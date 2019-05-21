from optuna.samplers import BaseSampler
from optuna.samplers import TPESampler

try:
    import skopt
    _available = True
except ImportError as e:
    _import_error = e
    # SkoptSampler is disabled because Scikit-Optimize is not available.
    _available = False

if types.TYPE_CHECKING:
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.study import RunningStudy  # NOQA
    from optuna.trial import FrozenTrial  # NOQA


class SkoptSampler(BaseSampler):
    def __init__(self, independent_sampler=None):
        # type: (Optional[BaseSampler]) -> None

        _check_skopt_availability()

        self.optimizer = None
        self.search_space = {}
        self.param_names = []
        self.known_trials = set()
        self.independent_sampler = independent_sampler or TPESampler()

    def sample_relative(self, study, trial, search_space):
        # type: (RunningStudy, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, float]

        if len(search_space) == 0:
            return {}

        raise NotImplementedError

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (RunningStudy, FrozenTrial, str, BaseDistribution) -> float

        return self.independent_sampler.sample_independent(study, trial, param_name,
                                                           param_distribution)


def _check_skopt_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'Scikit-Optimize is not available. Please install Scikit-Optimize to use this feature. '
            'Scikit-Optimize can be installed by executing `$ pip install skopt`. '
            'For further information, please refer to the installation guide of Scikit-Optimize. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
