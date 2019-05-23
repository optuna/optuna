import math

from optuna.pruners import BasePruner
from optuna.storages import BaseStorage  # NOQA
from optuna.structs import StudyDirection
from optuna.structs import TrialState


class PercentilePruner(BasePruner):
    """Like :class:`~optuna.pruners.MedianPruner`, but prunes if the best
    intermediate result is worse than top of specified percentile of intermediate
    results of previous trials at the same step.

    Example:

        .. code::

            >>> from optuna import create_study
            >>> from optuna.pruners import PercentilePruner
            >>>
            >>> def objective(trial):
            >>>     ...
            >>>
            >>> study = create_study(pruner=PercentilePruner())
            >>> study.optimize(objective)

    Args:
        n_startup_trials:
            Pruning is disabled until the given number of trials finish in the same study.
        n_warmup_steps:
            Pruning is disabled until the trial reaches the given number of step.
        percentile:
            Percentile value which must be between 0 and 100 inclusive
            (ex: Top of 25th percentile trials are kept when given 25.0).
    """

    def __init__(self, n_startup_trials=5, n_warmup_steps=0, percentile=25.0):
        # type: (int, int, float) -> None

        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.percentile = percentile

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

        p = storage.get_percentile_intermediate_result_over_trials(study_id, step, self.percentile)
        if math.isnan(p):
            return False

        if storage.get_study_direction(study_id) == StudyDirection.MAXIMIZE:
            return best_intermediate_result < p
        return best_intermediate_result > p
