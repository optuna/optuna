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
        """Sample a parameter based on the previous trials and the given distribution.

        Note that this method is not supposed to be called by library users. Instead,
        :class:`optuna.trial.Trial` provides user interfaces to sample parameters in an objective
        function.

        Args:
            trial:
                Trial object that contains the information of the target trial.
            param_name:
                Name of the sampled parameter.
            param_distribution:
                Distribution object that specifies a prior and/or scale of the sampling algorithm.

        Returns:
            A float value in the internal representation of Optuna.

        """

        raise NotImplementedError

    def before_trial(self, trial):
        # type: (FrozenTrial) -> None
        """A callback method that invoked each time before the target objective function is called.

        Args:
            trial:
                Trial object that contains the information of the target trial.
                The state of the trial object always be :obj:`~optuna.structs.TrialState.RUNNING`.

        """

        pass

    def after_trial(self, trial):
        # type: (FrozenTrial) -> None
        """A callback method that invoked each time after the target objective function is called.

        Args:
            trial:
                Trial object that contains the information of the target trial.
                The state of the trial object never be :obj:`~optuna.structs.TrialState.RUNNING`.

        """

        pass

    @property
    def study(self):
        # type: () -> RunningStudy
        """Return the target study."""

        if not hasattr(self, '_study'):
            raise RuntimeError('`_study` field has not yet been set.')

        return self._study

    def _set_study(self, study):
        # type: (RunningStudy) -> None

        self._study = study
