import numpy as np

from optuna import distributions
from optuna import logging
from optuna.samplers import BaseSampler
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
from optuna.structs import StudyDirection
from optuna.structs import TrialState
from optuna import types

try:
    import skopt
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
    from optuna.study import RunningStudy  # NOQA


class SkoptSampler(BaseSampler):
    def __init__(self, base_estimator="GP", independent_sampler=None):
        # type: (Union[str, RegressorMixin], Optional[BaseSampler]) -> None

        # TODO(ohta): Add other skopt options
        # See: https://scikit-optimize.github.io/optimizer/index.html#skopt.optimizer.Optimizer

        _check_skopt_availability()

        self.base_estimator = base_estimator
        self.independent_sampler = independent_sampler or TPESampler()
        self.random_sampler = RandomSampler()

        self.optimizer = None  # type: skopt.Optimizer
        self.search_space = {}  # type: Dict[str, distributions.BaseDistribution]
        self.param_names = []  # type: List[str]
        self.known_trials = set()  # type: Set[int]
        self.logger = logging.get_logger(__name__)

    def sample_relative(self, study, trial, search_space):
        # type: (RunningStudy, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, float]

        if len(search_space) == 0:
            return {}

        if self.search_space != search_space:
            self.logger.debug("Search space changed: {} => {}".format(
                self.search_space, search_space))
            self._initialize_optimizer(search_space)

        self._tell_unknown_trials(study)

        params = {}
        param_values = self.optimizer.ask()
        for name, value in zip(self.param_names, param_values):
            distribution = self.search_space[name]
            if isinstance(distribution, distributions.DiscreteUniformDistribution):
                value = value * distribution.q + distribution.low

            params[name] = distribution.to_internal_repr(value)

        return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (RunningStudy, FrozenTrial, str, BaseDistribution) -> float

        return self.independent_sampler.sample_independent(study, trial, param_name,
                                                           param_distribution)

    def _initialize_optimizer(self, search_space):
        # type: (Dict[str, BaseDistribution]) -> None

        self.search_space = search_space
        self.param_names = []
        self.known_trials = set()

        dimensions = []  # type: List[Any]
        for name, distribution in search_space.items():
            if isinstance(distribution, distributions.UniformDistribution):
                # Convert to half-closed range.
                # See: https://scikit-optimize.github.io/space/space.m.html#skopt.space.space.Real
                high = max(distribution.low, np.nextafter(distribution.high,  float('-inf')))

                dimensions.append((float(distribution.low), float(high)))
                self.param_names.append(name)
            elif isinstance(distribution, distributions.LogUniformDistribution):
                high = max(distribution.low, np.nextafter(distribution.high,  float('-inf')))

                dimensions.append((float(distribution.low), float(high), 'log-uniform'))
                self.param_names.append(name)
            elif isinstance(distribution, distributions.IntUniformDistribution):
                dimensions.append((int(distribution.low), int(distribution.high)))
                self.param_names.append(name)
            elif isinstance(distribution, distributions.DiscreteUniformDistribution):
                count = (distribution.high - distribution.low) // distribution.q
                dimensions.append((0, count))
                self.param_names.append(name)
            elif isinstance(distribution, distributions.CategoricalDistribution):
                dimensions.append(list(distribution.choices))
                self.param_names.append(name)
            else:
                raise NotImplementedError(
                    "The distribution {} is not implemented.".format(distribution))

        self.optimizer = skopt.Optimizer(dimensions, self.base_estimator)

    def _tell_unknown_trials(self, study):
        # type: (RunningStudy) -> None

        xs = []
        ys = []
        for trial in study.trials:
            if trial.number in self.known_trials or trial.state != TrialState.COMPLETE:
                continue
            self.known_trials.add(trial.number)

            result = self._to_skopt_observation(study, trial)
            if result is not None:
                x, y = result
                xs.append(x)
                ys.append(y)

        self.optimizer.tell(xs, ys)

    def _to_skopt_observation(self, study, trial):
        # type: (RunningStudy, FrozenTrial) -> Optional[Tuple[List[Any], float]]

        param_values = []
        for name in self.param_names:
            distribution = self.search_space[name]

            if name in trial.params:
                param_value = trial.params[name]
                param_internal_value = distribution.to_internal_repr(param_value)
                if not distribution._contains(param_internal_value):
                    return None
            else:
                param_value = self.random_sampler.sample_independent(study, trial, name,
                                                                     distribution)

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
