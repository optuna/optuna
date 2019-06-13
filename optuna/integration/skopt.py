from __future__ import absolute_import

import numpy as np

from optuna import distributions
from optuna.samplers import BaseSampler
from optuna.samplers import TPESampler
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
    from typing import Set  # NOQA
    from typing import Tuple  # NOQA
    from typing import Union  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import InTrialStudy  # NOQA


class SkoptSampler(BaseSampler):
    """Sampler using Scikit-Optimize as the backend.

    Example:

        Optimize a simple quadratic function by using :class:`~optuna.integration.SkoptSampler`.

        .. code::

                def objective(trial):
                    x = trial.suggest_uniform('x', -100, 100)
                    y = trial.suggest_int('y', -10, 10)
                    return x**2 + y

                sampler = optuna.integration.SkoptSampler()
                study = optuna.create_study(sampler=sampler)
                study.optimize(objective, n_trials=100)

    Args:
        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independently
            sampling parameters that unknown to :class:`~optuna.integration.SkoptSampler`.
            An "unknown parameter" means a parameter that isn't contained in
            :meth:`~optuna.study.InTrialStudy.product_search_space` of the target study.

            If :obj:`None` is specified, :class:`~optuna.samplers.TPESampler` is used
            as the default. See also :class:`~optuna.samplers`.
        skopt_kwargs:
            Keyword arguments passed to the constructor of
            `skopt.Optimizer <https://scikit-optimize.github.io/#skopt.Optimizer>`_
            class.

            Note that the ``dimensions`` argument is automatically added by
            :class:`~optuna.integration.SkoptSampler`, so you don't have to specify it
            (it will be ignored even if specified).

    """

    def __init__(self, independent_sampler=None, skopt_kwargs=None):
        # type: (Optional[BaseSampler], Optional[Dict[str, Any]]) -> None

        _check_skopt_availability()

        self._skopt_kwargs = skopt_kwargs or {}
        if 'dimensions' in self._skopt_kwargs:
            del self._skopt_kwargs['dimensions']

        self._independent_sampler = independent_sampler or TPESampler()

    def infer_relative_search_space(self, study, trial):
        # type: (InTrialStudy, FrozenTrial) -> Dict[str, BaseDistribution]

        search_space = {}
        for name, distribution in study.product_search_space.items():
            # Skip if the range of the distribution is empty.
            if isinstance(distribution, distributions.UniformDistribution):
                if distribution.low == distribution.high:
                    continue
            elif isinstance(distribution, distributions.LogUniformDistribution):
                if distribution.low == distribution.high:
                    continue
            elif isinstance(distribution, distributions.IntUniformDistribution):
                if distribution.low == distribution.high:
                    continue
            elif isinstance(distribution, distributions.DiscreteUniformDistribution):
                if distribution.low == distribution.high:
                    continue

            search_space[name] = distribution

        return search_space

    def sample_relative(self, study, trial, search_space):
        # type: (InTrialStudy, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, float]

        if len(search_space) == 0:
            return {}

        optimizer = _Optimizer(search_space, self._skopt_kwargs)
        optimizer.tell(study)
        return optimizer.ask()

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (InTrialStudy, FrozenTrial, str, BaseDistribution) -> float

        return self._independent_sampler.sample_independent(study, trial, param_name,
                                                            param_distribution)


class _Optimizer(object):
    def __init__(self, search_space, skopt_kwargs=None):
        # type: (Dict[str, BaseDistribution], Optional[Dict[str, Any]]) -> None

        self._search_space = search_space

        dimensions = []  # type: List[Any]
        for name, distribution in sorted(self._search_space.items()):
            if isinstance(distribution, distributions.UniformDistribution):
                high = max(distribution.low, np.nextafter(distribution.high, float('-inf')))
                dimension = space.Real(distribution.low, high)
            elif isinstance(distribution, distributions.LogUniformDistribution):
                high = max(distribution.low, np.nextafter(distribution.high, float('-inf')))
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
            if not trial.state.is_finished():
                continue

            result = self._trial_to_skopt_observation(study, trial)
            if result is not None:
                x, y = result
                xs.append(x)
                ys.append(y)

        self._optimizer.tell(xs, ys)

    def ask(self):
        # type: () -> Dict[str, float]

        params = {}
        param_values = self._optimizer.ask()
        for (name, distribution), value in zip(sorted(self._search_space.items()), param_values):
            if isinstance(distribution, distributions.DiscreteUniformDistribution):
                value = value * distribution.q + distribution.low

            params[name] = distribution.to_internal_repr(value)

        return params

    def _trial_to_skopt_observation(self, study, trial):
        # type: (InTrialStudy, FrozenTrial) -> Optional[Tuple[List[Any], float]]

        param_values = []
        for name, distribution in sorted(self._search_space.items()):
            if name not in trial.params:
                return None

            param_value = trial.params[name]
            param_internal_value = distribution.to_internal_repr(param_value)
            if not distribution._contains(param_internal_value):
                return None

            if isinstance(distribution, distributions.DiscreteUniformDistribution):
                param_value = (param_value - distribution.low) // distribution.q

            param_values.append(param_value)

        value = trial.value
        if value is None:
            return None

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
