import functools
import math

import numpy as np

from optuna.pruners import BasePruner
from optuna.study import StudyDirection
from optuna.trial._state import TrialState
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import KeysView  # NOQA
    from typing import List  # NOQA

    from optuna.study import Study  # NOQA
    from optuna.trial import FrozenTrial  # NOQA


def _get_best_intermediate_result_over_steps(trial, direction):
    # type: (FrozenTrial, StudyDirection) -> float

    values = np.array(list(trial.intermediate_values.values()), np.float)
    if direction == StudyDirection.MAXIMIZE:
        return np.nanmax(values)
    return np.nanmin(values)


def _get_percentile_intermediate_result_over_trials(all_trials, direction, step, percentile):
    # type: (List[FrozenTrial], StudyDirection, int, float) -> float

    completed_trials = [t for t in all_trials if t.state == TrialState.COMPLETE]

    if len(completed_trials) == 0:
        raise ValueError("No trials have been completed.")

    if direction == StudyDirection.MAXIMIZE:
        percentile = 100 - percentile

    return float(
        np.nanpercentile(
            np.array(
                [
                    t.intermediate_values[step]
                    for t in completed_trials
                    if step in t.intermediate_values
                ],
                np.float,
            ),
            percentile,
        )
    )


def _is_first_in_interval_step(step, intermediate_steps, n_warmup_steps, interval_steps):
    # type: (int, KeysView[int], int, int) -> bool

    nearest_lower_pruning_step = (
        (step - n_warmup_steps - 1) // interval_steps * interval_steps + n_warmup_steps + 1
    )
    assert nearest_lower_pruning_step >= 0

    # `intermediate_steps` may not be sorted so we must go through all elements.
    second_last_step = functools.reduce(
        lambda second_last_step, s: s if s > second_last_step and s != step else second_last_step,
        intermediate_steps,
        -1,
    )

    return second_last_step < nearest_lower_pruning_step


class PercentilePruner(BasePruner):
    """Pruner to keep the specified percentile of the trials.

    Prune if the best intermediate value is in the bottom percentile among trials at the same step.

    Example:

        .. testcode::

            import numpy as np
            from sklearn.datasets import load_iris
            from sklearn.linear_model import SGDClassifier
            from sklearn.model_selection import train_test_split

            import optuna

            X, y = load_iris(return_X_y=True)
            X_train, X_valid, y_train, y_valid = train_test_split(X, y)
            classes = np.unique(y)

            def objective(trial):
                alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
                clf = SGDClassifier(alpha=alpha)
                n_train_iter = 100

                for step in range(n_train_iter):
                    clf.partial_fit(X_train, y_train, classes=classes)

                    intermediate_value = clf.score(X_valid, y_valid)
                    trial.report(intermediate_value, step)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return clf.score(X_valid, y_valid)

            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.PercentilePruner(25.0, n_startup_trials=5,
                                                       n_warmup_steps=30, interval_steps=10))
            study.optimize(objective, n_trials=20)

    Args:
        percentile:
            Percentile which must be between 0 and 100 inclusive
            (e.g., When given 25.0, top of 25th percentile trials are kept).
        n_startup_trials:
            Pruning is disabled until the given number of trials finish in the same study.
        n_warmup_steps:
            Pruning is disabled until the trial exceeds the given number of step.
        interval_steps:
            Interval in number of steps between the pruning checks, offset by the warmup steps.
            If no value has been reported at the time of a pruning check, that particular check
            will be postponed until a value is reported. Value must be at least 1.
    """

    def __init__(self, percentile, n_startup_trials=5, n_warmup_steps=0, interval_steps=1):
        # type: (float, int, int, int) -> None

        if not 0.0 <= percentile <= 100:
            raise ValueError(
                "Percentile must be between 0 and 100 inclusive but got {}.".format(percentile)
            )
        if n_startup_trials < 0:
            raise ValueError(
                "Number of startup trials cannot be negative but got {}.".format(n_startup_trials)
            )
        if n_warmup_steps < 0:
            raise ValueError(
                "Number of warmup steps cannot be negative but got {}.".format(n_warmup_steps)
            )
        if interval_steps < 1:
            raise ValueError(
                "Pruning interval steps must be at least 1 but got {}.".format(interval_steps)
            )

        self._percentile = percentile
        self._n_startup_trials = n_startup_trials
        self._n_warmup_steps = n_warmup_steps
        self._interval_steps = interval_steps

    def prune(self, study, trial):
        # type: (Study, FrozenTrial) -> bool

        all_trials = study.get_trials(deepcopy=False)
        n_trials = len([t for t in all_trials if t.state == TrialState.COMPLETE])

        if n_trials == 0:
            return False

        if n_trials < self._n_startup_trials:
            return False

        step = trial.last_step
        if step is None:
            return False

        n_warmup_steps = self._n_warmup_steps
        if step <= n_warmup_steps:
            return False

        if not _is_first_in_interval_step(
            step, trial.intermediate_values.keys(), n_warmup_steps, self._interval_steps
        ):
            return False

        direction = study.direction
        best_intermediate_result = _get_best_intermediate_result_over_steps(trial, direction)
        if math.isnan(best_intermediate_result):
            return True

        p = _get_percentile_intermediate_result_over_trials(
            all_trials, direction, step, self._percentile
        )
        if math.isnan(p):
            return False

        if direction == StudyDirection.MAXIMIZE:
            return best_intermediate_result < p
        return best_intermediate_result > p
