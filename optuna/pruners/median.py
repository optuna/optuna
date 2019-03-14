import math

from optuna.pruners import BasePruner
from optuna.storages import BaseStorage  # NOQA
from optuna.structs import StudyDirection
from optuna.structs import TrialState


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

        n_trials = storage.get_n_trials(study_id, TrialState.COMPLETE)

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

        if storage.get_study_direction(study_id) == StudyDirection.MAXIMIZE:
            return best_intermediate_result < median
        return best_intermediate_result > median
