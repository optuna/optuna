import abc
import six

from optuna.type_checking import TYPE_CHECKING

if TYPE_CHECKING:
    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import Study  # NOQA


@six.add_metaclass(abc.ABCMeta)
class BasePruner(object):
    """Base class for pruners."""

    @abc.abstractmethod
    def prune(self, study, trial):
        # type: (Study, FrozenTrial) -> bool
        """Judge whether the trial should be pruned at the given step.

        Note that this method is not supposed to be called by library users. Instead,
        :func:`optuna.trial.Trial.report` and :func:`optuna.trial.Trial.should_prune` provide
        user interfaces to implement pruning mechanism in an objective function.

        Args:
            study:
                Study object of the target study.
            trial:
                FrozenTrial object of the target trial.

        Returns:
            A boolean value representing whether the trial should be pruned.
        """

        raise NotImplementedError
