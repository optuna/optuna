import abc

from optuna.type_checking import TYPE_CHECKING

if TYPE_CHECKING:
    from optuna.study import Study  # NOQA
    from optuna.trial import FrozenTrial  # NOQA


class BasePruner(object, metaclass=abc.ABCMeta):
    """Base class for pruners."""

    @abc.abstractmethod
    def prune(self, study, trial):
        # type: (Study, FrozenTrial) -> bool
        """Judge whether the trial should be pruned based on the reported values.

        Note that this method is not supposed to be called by library users. Instead,
        :func:`optuna.trial.Trial.report` and :func:`optuna.trial.Trial.should_prune` provide
        user interfaces to implement pruning mechanism in an objective function.

        Args:
            study:
                Study object of the target study.
            trial:
                FrozenTrial object of the target trial.
                Take a copy before modifying this object.

        Returns:
            A boolean value representing whether the trial should be pruned.
        """

        raise NotImplementedError
