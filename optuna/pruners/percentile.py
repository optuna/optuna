import math
import numpy as np

from optuna.pruners import BasePruner
from optuna.structs import StudyDirection
from optuna.structs import TrialState
from optuna import structs
from optuna import types

if types.TYPE_CHECKING:
    from optuna.storages import BaseStorage  # NOQA


def get_best_intermediate_result_over_steps(trial, direction):
    # type: (structs.FrozenTrial, structs.StudyDirection) -> float

    values = np.array(list(trial.intermediate_values.values()), np.float)
    if direction == structs.StudyDirection.MAXIMIZE:
        return np.nanmax(values)
    return np.nanmin(values)


class PercentilePruner(BasePruner):
    """Pruner to keep the specified percentile of the trials.

    Prune if the best intermediate value is in the bottom percentile among trials at the same step.

    Example:

        .. code::

            >>> from optuna import create_study
            >>> from optuna.pruners import PercentilePruner
            >>>
            >>> def objective(trial):
            >>>     ...
            >>>
            >>> study = create_study(pruner=PercentilePruner(25.0))
            >>> study.optimize(objective)

    Args:
        percentile:
            Percentile which must be between 0 and 100 inclusive
            (e.g., When given 25.0, top of 25th percentile trials are kept).
        n_startup_trials:
            Pruning is disabled until the given number of trials finish in the same study.
        n_warmup_steps:
            Pruning is disabled until the trial reaches the given number of step.
    """

    def __init__(self, percentile, n_startup_trials=5, n_warmup_steps=0):
        # type: (float, int, int) -> None

        self.percentile = percentile
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

        trial = storage.get_trial(trial_id)
        if len(trial.intermediate_values) == 0:
            return False

        direction = storage.get_study_direction(study_id)
        best_intermediate_result = get_best_intermediate_result_over_steps(trial, direction)
        if math.isnan(best_intermediate_result):
            return True

        p = storage.get_percentile_intermediate_result_over_trials(study_id, step, self.percentile)
        if math.isnan(p):
            return False

        if direction == StudyDirection.MAXIMIZE:
            return best_intermediate_result < p
        return best_intermediate_result > p
