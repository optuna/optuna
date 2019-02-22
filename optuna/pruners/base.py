import abc
import six

from optuna.storages import BaseStorage  # NOQA


@six.add_metaclass(abc.ABCMeta)
class BasePruner(object):
    """Base class for pruners."""

    @abc.abstractmethod
    def prune(self, storage, study_id, trial_id, step):
        # type: (BaseStorage, int, int, int) -> bool
        """Judge whether the trial should be pruned at the given step.

        Note that this method is not supposed to be called by library users. Instead,
        :func:`optuna.trial.Trial.report` and :func:`optuna.trial.Trial.should_prune` provide
        user interfaces to implement pruning mechanism in an objective function.

        Args:
            storage:
                Storage object.
            study_id:
                Identifier of the target study.
            trial_id:
                Identifier of the target trial.
            step:
                Step number.

        Returns:
            A boolean value representing whether the trial should be pruned.
        """

        raise NotImplementedError
