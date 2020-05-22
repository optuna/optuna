import numpy as np

import optuna
from optuna import distributions
from optuna import samplers
from optuna.samplers import BaseSampler
from optuna.study import StudyDirection
from optuna.trial import TrialState
from optuna import type_checking

try:
    import skopt
    from skopt.space import space

    _available = True
except ImportError as e:
    _import_error = e
    # SkoptSampler is disabled because Scikit-Optimize is not available.
    _available = False

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.study import Study  # NOQA
    from optuna.trial import FrozenTrial  # NOQA


class SkoptSampler(BaseSampler):
    """Sampler using Scikit-Optimize as the backend.

    Example:

        Optimize a simple quadratic function by using :class:`~optuna.integration.SkoptSampler`.

        .. testcode::

                import optuna

                def objective(trial):
                    x = trial.suggest_uniform('x', -10, 10)
                    y = trial.suggest_int('y', 0, 10)
                    return x**2 + y

                sampler = optuna.integration.SkoptSampler()
                study = optuna.create_study(sampler=sampler)
                study.optimize(objective, n_trials=10)

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

        n_startup_trials:
            The independent sampling is used until the given number of trials finish in the
            same study.
    """

    def __init__(
        self,
        independent_sampler=None,
        warn_independent_sampling=True,
        skopt_kwargs=None,
        n_startup_trials=1,
    ):
        # type: (Optional[BaseSampler], bool, Optional[Dict[str, Any]], int) -> None

        _check_skopt_availability()

        self._skopt_kwargs = skopt_kwargs or {}
        if "dimensions" in self._skopt_kwargs:
            del self._skopt_kwargs["dimensions"]

        self._independent_sampler = independent_sampler or samplers.RandomSampler()
        self._warn_independent_sampling = warn_independent_sampling
        self._n_startup_trials = n_startup_trials
        self._search_space = samplers.IntersectionSearchSpace()

    def reseed_rng(self) -> None:

        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(self, study, trial):
        # type: (Study, FrozenTrial) -> Dict[str, BaseDistribution]

        search_space = {}
        for name, distribution in self._search_space.calculate(study).items():
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
        # type: (Study, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, Any]

        if len(search_space) == 0:
            return {}

        complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if len(complete_trials) < self._n_startup_trials:
            return {}

        optimizer = _Optimizer(search_space, self._skopt_kwargs)
        optimizer.tell(study, complete_trials)
        return optimizer.ask()

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (Study, FrozenTrial, str, BaseDistribution) -> Any

        if self._warn_independent_sampling:
            complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
            if len(complete_trials) >= self._n_startup_trials:
                self._log_independent_sampling(trial, param_name)

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _log_independent_sampling(self, trial, param_name):
        # type: (FrozenTrial, str) -> None

        logger = optuna.logging.get_logger(__name__)
        logger.warning(
            "The parameter '{}' in trial#{} is sampled independently "
            "by using `{}` instead of `SkoptSampler` "
            "(optimization performance may be degraded). "
            "You can suppress this warning by setting `warn_independent_sampling` "
            "to `False` in the constructor of `SkoptSampler`, "
            "if this independent sampling is intended behavior.".format(
                param_name, trial.number, self._independent_sampler.__class__.__name__
            )
        )


class _Optimizer(object):
    def __init__(self, search_space, skopt_kwargs):
        # type: (Dict[str, BaseDistribution], Dict[str, Any]) -> None

        self._search_space = search_space

        dimensions = []
        for name, distribution in sorted(self._search_space.items()):
            # TODO(nzw0301) support IntLogUniform
            if isinstance(distribution, distributions.UniformDistribution):
                # Convert the upper bound from exclusive (optuna) to inclusive (skopt).
                high = np.nextafter(distribution.high, float("-inf"))
                dimension = space.Real(distribution.low, high)
            elif isinstance(distribution, distributions.LogUniformDistribution):
                # Convert the upper bound from exclusive (optuna) to inclusive (skopt).
                high = np.nextafter(distribution.high, float("-inf"))
                dimension = space.Real(distribution.low, high, prior="log-uniform")
            elif isinstance(distribution, distributions.IntUniformDistribution):
                count = (distribution.high - distribution.low) // distribution.step
                dimension = space.Integer(0, count)
            elif isinstance(distribution, distributions.DiscreteUniformDistribution):
                count = int((distribution.high - distribution.low) // distribution.q)
                dimension = space.Integer(0, count)
            elif isinstance(distribution, distributions.CategoricalDistribution):
                dimension = space.Categorical(distribution.choices)
            else:
                raise NotImplementedError(
                    "The distribution {} is not implemented.".format(distribution)
                )

            dimensions.append(dimension)

        self._optimizer = skopt.Optimizer(dimensions, **skopt_kwargs)

    def tell(self, study, complete_trials):
        # type: (Study, List[FrozenTrial]) -> None

        xs = []
        ys = []

        for trial in complete_trials:
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
            if isinstance(distribution, distributions.IntUniformDistribution):
                value = value * distribution.step + distribution.low

            params[name] = value

        return params

    def _is_compatible(self, trial):
        # type: (FrozenTrial) -> bool

        # Thanks to `intersection_search_space()` function, in sequential optimization,
        # the parameters of complete trials are always compatible with the search space.
        #
        # However, in distributed optimization, incompatible trials may complete on a worker
        # just after an intersection search space is calculated on another worker.

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
        # type: (Study, FrozenTrial) -> Tuple[List[Any], float]

        param_values = []
        for name, distribution in sorted(self._search_space.items()):
            param_value = trial.params[name]

            if isinstance(distribution, distributions.DiscreteUniformDistribution):
                param_value = (param_value - distribution.low) // distribution.q
            if isinstance(distribution, distributions.IntUniformDistribution):
                param_value = (param_value - distribution.low) // distribution.step

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
            "Scikit-Optimize is not available. Please install it to use this feature. "
            "Scikit-Optimize can be installed by executing `$ pip install scikit-optimize`. "
            "For further information, please refer to the installation guide of Scikit-Optimize. "
            "(The actual import error is as follows: " + str(_import_error) + ")"
        )
