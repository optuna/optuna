from collections import OrderedDict

from optuna import distributions
from optuna import logging
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
    from sklearn.base import RegressorMixin  # NOQA
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
    def __init__(self, independent_sampler=None, skopt_kwargs=None):
        # type: (Optional[BaseSampler], Optional[Dict[str, Any]]) -> None

        _check_skopt_availability()

        self.skopt_kwargs = skopt_kwargs or {}
        self.independent_sampler = independent_sampler or TPESampler()
        self.logger = logging.get_logger(__name__)

    def infer_relative_search_space(self, study, trial):
        # type: (InTrialStudy, FrozenTrial) -> Dict[str, BaseDistribution]

        search_space = {}
        for name, distribution in study.product_search_space.items():
            if isinstance(distribution, distributions.UniformDistribution):
                if distribution.low == distribution.high:
                    continue
            elif isinstance(distribution, distributions.LogUniformDistribution):
                if distribution.low == distribution.high:
                    continue

            search_space[name] = distribution

        return search_space

    def sample_relative(self, study, trial, search_space):
        # type: (InTrialStudy, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, float]

        if len(search_space) == 0:
            return {}

        optimizer = _Optimizer(search_space, self.skopt_kwargs)
        optimizer.tell(study)
        return optimizer.ask()

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (InTrialStudy, FrozenTrial, str, BaseDistribution) -> float

        return self.independent_sampler.sample_independent(study, trial, param_name,
                                                           param_distribution)


class _Optimizer(object):
    def __init__(self, search_space, skopt_kwargs=None):
        # type: (Dict[str, BaseDistribution], Optional[Dict[str, Any]]) -> None

        self.search_space = OrderedDict(search_space)

        dimensions = []  # type: List[Any]
        for name, distribution in self.search_space.items():
            if isinstance(distribution, distributions.UniformDistribution):
                dimension = space.Real(distribution.low, distribution.high, name=name)
            elif isinstance(distribution, distributions.LogUniformDistribution):
                dimension = space.Real(distribution.low,
                                       distribution.high,
                                       prior='log-uniform',
                                       name=name)
            elif isinstance(distribution, distributions.IntUniformDistribution):
                dimension = space.Integer(distribution.low, distribution.high + 1, name=name)
            elif isinstance(distribution, distributions.DiscreteUniformDistribution):
                count = (distribution.high - distribution.low) // distribution.q
                dimension = space.Integer(0, count + 1, name=name)
            elif isinstance(distribution, distributions.CategoricalDistribution):
                dimension = space.Categorical(distribution.choices, name=name)
            else:
                raise NotImplementedError(
                    "The distribution {} is not implemented.".format(distribution))

            dimensions.append(dimension)

        self.optimizer = skopt.Optimizer(dimensions, **skopt_kwargs)

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

        self.optimizer.tell(xs, ys)

    def ask(self):
        # type: () -> Dict[str, float]

        params = {}
        param_values = self.optimizer.ask()
        for (name, distribution), value in zip(self.search_space.items(), param_values):
            if isinstance(distribution, distributions.DiscreteUniformDistribution):
                value = value * distribution.q + distribution.low

            params[name] = distribution.to_internal_repr(value)

        return params

    def _trial_to_skopt_observation(self, study, trial):
        # type: (InTrialStudy, FrozenTrial) -> Optional[Tuple[List[Any], float]]

        param_values = []
        for name, distribution in self.search_space.items():
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
