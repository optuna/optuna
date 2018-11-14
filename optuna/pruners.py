import abc
import math
import six

from optuna.storages import BaseStorage  # NOQA
from optuna.structs import TrialState


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


class MedianPruner(BasePruner):

    """Pruner using the median stopping rule.

    Prune if the trial's best intermediate result is worse than median of intermediate results of
    previous trials at the same step.

    Example:

        We minimize an objective function with the median stopping rule.

        .. code::

            >>> from optuna import create_study
            >>> from optuna.pruners import MedianPruner
            >>>
            >>> def objective(trial):
            >>>     ...
            >>>
            >>> study = create_study(pruner=MedianPruner())
            >>> study.optimize(objective)

    Args:
        n_startup_trials:
            Pruning is disabled until the given number of trials finish in the same study.
        n_warmup_steps:
            Pruning is disabled until the trial reaches the given number of step.
    """

    def __init__(self, n_startup_trials=5, n_warmup_steps=0):
        # type: (int, int) -> None

        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps

    def prune(self, storage, study_id, trial_id, step):
        # type: (BaseStorage, int, int, int) -> bool

        """Please consult the documentation for :func:`BasePruner.prune`."""

        # TODO(Yanase): Implement a method of storage to just retrieve the number of trials.
        n_trials = len([t for t in storage.get_all_trials(study_id)
                        if t.state == TrialState.COMPLETE])

        if n_trials == 0:
            return False

        if n_trials < self.n_startup_trials:
            return False

        if step <= self.n_warmup_steps:
            return False

        if len(storage.get_trial(trial_id).intermediate_values) == 0:
            return False

        best_intermediate_result = storage.get_best_intermediate_result_over_steps(trial_id)
        if math.isnan(best_intermediate_result):
            return True

        median = storage.get_median_intermediate_result_over_trials(study_id, step)
        if math.isnan(median):
            return False

        return best_intermediate_result > median
