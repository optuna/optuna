import abc
import six

from optuna.distributions import BaseDistribution  # NOQA
from optuna.storages.base import BaseStorage  # NOQA
from optuna.structs import FrozenTrial  # NOQA
from optuna import types

if types.TYPE_CHECKING:
    from optuna.study import RunningStudy  # NOQA


@six.add_metaclass(abc.ABCMeta)
class BaseSampler(object):
    """Base class for samplers."""

    @abc.abstractmethod
    def sample(self, trial, param_name, param_distribution):
        # type: (FrozenTrial, str, BaseDistribution) -> float

        raise NotImplementedError

    def before_trial(self, trial):
        # type: (FrozenTrial) -> None

        pass

    def after_trial(self, trial):
        # type: (FrozenTrial) -> None

        pass

    @property
    def study(self):
        # type: () -> RunningStudy

        if not hasattr(self, '_study'):
            raise RuntimeError()

        return self._study

    def _set_study(self, study):
        # type: (RunningStudy) -> None

        self._study = study
