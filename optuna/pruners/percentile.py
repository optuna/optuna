import math
import numpy as np

from optuna.pruners import BasePruner
from optuna import structs
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import List  # NOQA

    from optuna.study import Study  # NOQA


def _get_best_intermediate_result_over_steps(trial, direction):
    # type: (structs.FrozenTrial, structs.StudyDirection) -> float

    values = np.array(list(trial.intermediate_values.values()), np.float)
    if direction == structs.StudyDirection.MAXIMIZE:
        return np.nanmax(values)
    return np.nanmin(values)


def _get_percentile_intermediate_result_over_trials(all_trials, direction, step, percentile):
    # type: (List[structs.FrozenTrial], structs.StudyDirection, int, float) -> float

    completed_trials = [t for t in all_trials if t.state == structs.TrialState.COMPLETE]

    if len(completed_trials) == 0:
        raise ValueError("No trials have been completed.")

    if direction == structs.StudyDirection.MAXIMIZE:
        percentile = 100 - percentile

    return float(
        np.nanpercentile(
            np.array([
                t.intermediate_values[step]
                for t in completed_trials if step in t.intermediate_values
            ], np.float),
            percentile))


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

    def prune(self, study, trial):
        # type: (Study, structs.FrozenTrial) -> bool
        """Please consult the documentation for :func:`BasePruner.prune`."""

        all_trials = study.trials
        n_trials = len([t for t in all_trials
                        if t.state == structs.TrialState.COMPLETE])

        if n_trials == 0:
            return False

        if n_trials < self.n_startup_trials:
            return False

        step = trial.last_step
        if step is None:
            return False

        if step <= self.n_warmup_steps:
            return False

        if len(trial.intermediate_values) == 0:
            return False

        direction = study.direction
        best_intermediate_result = _get_best_intermediate_result_over_steps(trial, direction)
        if math.isnan(best_intermediate_result):
            return True

        p = _get_percentile_intermediate_result_over_trials(
            all_trials, direction, step, self.percentile)
        if math.isnan(p):
            return False

        if direction == structs.StudyDirection.MAXIMIZE:
            return best_intermediate_result < p
        return best_intermediate_result > p
