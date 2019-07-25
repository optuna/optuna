from __future__ import absolute_import

import numpy as np

import optuna
from optuna import distributions
from optuna import samplers
from optuna.samplers import BaseSampler
from optuna import structs
from optuna.structs import StudyDirection
from optuna import types

try:
    import skopt
    from skopt.space import space

    _available = True
except ImportError as e:
    _import_error = e
    # SkoptSampler is disabled because Scikit-Optimize is not available.
    _available = False

if types.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import InTrialStudy  # NOQA


class SkoptSampler(BaseSampler):
    """Sampler using Scikit-Optimize as the backend.

    Example:

        Optimize a simple quadratic function by using :class:`~optuna.integration.SkoptSampler`.

        .. code::

                def objective(trial):
                    x = trial.suggest_uniform('x', -10, 10)
                    y = trial.suggest_int('y', 0, 10)
                    return x**2 + y

                sampler = optuna.integration.SkoptSampler()
                study = optuna.create_study(sampler=sampler)
                study.optimize(objective, n_trials=100)

    Args:
        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler.
            The search space for :class:`~optuna.integration.SkoptSampler` is determined by
            :func:`~optuna.samplers.intersection_search_space()`.

            If :obj:`None` is specified, :class:`~optuna.samplers.RandomSampler` is used
            as the default.

            .. seealso::
                :class:`optuna.samplers` module provides built-in independent samplers
                such as :class:`~optuna.samplers.RandomSampler` and
                :class:`~optuna.samplers.TPESampler`.

        warn_independent_sampling:
            If this is :obj:`True`, a warning message is emitted when
            the value of a parameter is sampled by using an independent sampler.

            Note that the parameters of the first trial in a study are always sampled
            via an independent sampler, so no warning messages are emitted in this case.
        skopt_kwargs:
            Keyword arguments passed to the constructor of
            `skopt.Optimizer <https://scikit-optimize.github.io/#skopt.Optimizer>`_
            class.

            Note that ``dimensions`` argument in ``skopt_kwargs`` will be ignored
            because it is added by :class:`~optuna.integration.SkoptSampler` automatically.

    """

    def __init__(self, independent_sampler=None, warn_independent_sampling=True,
                 skopt_kwargs=None):
        # type: (Optional[BaseSampler], bool, Optional[Dict[str, Any]]) -> None

        _check_skopt_availability()

        self._skopt_kwargs = skopt_kwargs or {}
        if 'dimensions' in self._skopt_kwargs:
            del self._skopt_kwargs['dimensions']

        self._independent_sampler = independent_sampler or samplers.RandomSampler()
        self._warn_independent_sampling = warn_independent_sampling

    def infer_relative_search_space(self, study, trial):
        # type: (InTrialStudy, FrozenTrial) -> Dict[str, BaseDistribution]

        search_space = {}
        for name, distribution in samplers.intersection_search_space(study).items():
            if distribution.single():
                if not isinstance(distribution, distributions.CategoricalDistribution):
                    # `skopt` cannot handle non-categorical distributions that contain just
                    # a single value, so we skip this distribution.
                    #
                    # Note that `Trial` takes care of this distribution during suggestion.
                    continue

            search_space[name] = distribution

        return search_space

    def sample_relative(self, study, trial, search_space):
        # type: (InTrialStudy, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, Any]

        if len(search_space) == 0:
            return {}

        optimizer = _Optimizer(search_space, self._skopt_kwargs)
        optimizer.tell(study)
        return optimizer.ask()

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (InTrialStudy, FrozenTrial, str, BaseDistribution) -> Any

        if self._warn_independent_sampling:
            complete_trials = [t for t in study.trials if t.state == structs.TrialState.COMPLETE]
            if len(complete_trials) >= 1:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(study, trial, param_name,
                                                            param_distribution)

    def _log_independent_sampling(self, trial, param_name):
        # type: (FrozenTrial, str) -> None

        logger = optuna.logging.get_logger(__name__)
        logger.warning("The parameter '{}' in trial#{} is sampled independently "
                       "by using `{}` instead of `SkoptSampler` "
                       "(optimization performance may be degraded). "
                       "You can suppress this warning by setting `warn_independent_sampling` "
                       "to `False` in the constructor of `SkoptSampler`, "
                       "if this independent sampling is intended behavior.".format(
                           param_name, trial.number, self._independent_sampler.__class__.__name__))


class _Optimizer(object):
    def __init__(self, search_space, skopt_kwargs):
        # type: (Dict[str, BaseDistribution], Dict[str, Any]) -> None

        self._search_space = search_space

        dimensions = []
        for name, distribution in sorted(self._search_space.items()):
            if isinstance(distribution, distributions.UniformDistribution):
                # Convert the upper bound from exclusive (optuna) to inclusive (skopt).
                high = np.nextafter(distribution.high, float('-inf'))
                dimension = space.Real(distribution.low, high)
            elif isinstance(distribution, distributions.LogUniformDistribution):
                # Convert the upper bound from exclusive (optuna) to inclusive (skopt).
                high = np.nextafter(distribution.high, float('-inf'))
                dimension = space.Real(distribution.low, high, prior='log-uniform')
            elif isinstance(distribution, distributions.IntUniformDistribution):
                dimension = space.Integer(distribution.low, distribution.high)
            elif isinstance(distribution, distributions.DiscreteUniformDistribution):
                count = (distribution.high - distribution.low) // distribution.q
                dimension = space.Integer(0, count)
            elif isinstance(distribution, distributions.CategoricalDistribution):
                dimension = space.Categorical(distribution.choices)
            else:
                raise NotImplementedError(
                    "The distribution {} is not implemented.".format(distribution))

            dimensions.append(dimension)

        self._optimizer = skopt.Optimizer(dimensions, **skopt_kwargs)

    def tell(self, study):
        # type: (InTrialStudy) -> None

        xs = []
        ys = []
        for trial in study.trials:
            if trial.state != structs.TrialState.COMPLETE:
                continue

            if not self._is_compatible(trial):
                continue

            x, y = self._complete_trial_to_skopt_observation(study, trial)
            xs.append(x)
            ys.append(y)

        self._optimizer.tell(xs, ys)

    def ask(self):
        # type: () -> Dict[str, Any]

        params = {}
        param_values = self._optimizer.ask()
        for (name, distribution), value in zip(sorted(self._search_space.items()), param_values):
            if isinstance(distribution, distributions.DiscreteUniformDistribution):
                value = value * distribution.q + distribution.low

            params[name] = value

        return params

    def _is_compatible(self, trial):
        # type: (FrozenTrial) -> bool

        # Thanks to `intersection_search_space()` function, in sequential optimization,
        # the parameters of complete trials are always compatible with the search space.
        #
        # However, in distributed optimization, incompatible trials may complete on a worker
        # just after a product search space is calculated on another worker.

        for name, distribution in self._search_space.items():
            if name not in trial.params:
                return False

            distributions.check_distribution_compatibility(distribution, trial.distributions[name])
            param_value = trial.params[name]
            param_internal_value = distribution.to_internal_repr(param_value)
            if not distribution._contains(param_internal_value):
                return False

        return True

    def _complete_trial_to_skopt_observation(self, study, trial):
        # type: (InTrialStudy, FrozenTrial) -> Tuple[List[Any], float]

        param_values = []
        for name, distribution in sorted(self._search_space.items()):
            param_value = trial.params[name]

            if isinstance(distribution, distributions.DiscreteUniformDistribution):
                param_value = (param_value - distribution.low) // distribution.q

            param_values.append(param_value)

        value = trial.value
        assert value is not None

        if study.direction == StudyDirection.MAXIMIZE:
            value = -value

        return param_values, value


def _check_skopt_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'Scikit-Optimize is not available. Please install it to use this feature. '
            'Scikit-Optimize can be installed by executing `$ pip install scikit-optimize`. '
            'For further information, please refer to the installation guide of Scikit-Optimize. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
