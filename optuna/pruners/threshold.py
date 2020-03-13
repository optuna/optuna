import math
from typing import Optional

import optuna
from optuna.pruners import BasePruner
from optuna.pruners.percentile import _is_first_in_interval_step


class ThresholdPruner(BasePruner):
    """Pruner to detect outlying metrics of the trials.

    Prune if a metric exceeds upper bound threshold or
    falls behind lower bound threshold.

    Example:
        .. testcode::

            from optuna import create_study
            from optuna.exceptions import TrialPruned
            from optuna.pruners import ThresholdPruner


            def objective_for_upper(trial):
                for step, y in enumerate(ys_for_upper):
                    trial.report(y, step)

                    if trial.should_prune():
                        raise TrialPruned()
                return ys_for_upper[-1]


            def objective_for_lower(trial):
                for step, y in enumerate(ys_for_lower):
                    trial.report(y, step)

                    if trial.should_prune():
                        raise TrialPruned()
                return ys_for_lower[-1]


            ys_for_upper = [0.0, 0.1, 0.2, 0.5, 1.2]
            ys_for_lower = [100.0, 90.0, 0.1, 0.0, -1]
            n_trial_step = 5

            study = create_study(pruner=ThresholdPruner(upper=1.0))
            study.optimize(objective_1, n_trials=10)

            study = create_study(pruner=ThresholdPruner(lower=0.0))
            study.optimize(objective_2, n_trials=10)


    Args
        lower:
            A minimum value which determines whether pruner prunes or not.
            If an intermediate value is smaller than lower, it prunes.
        upper:
            maximum value which determines whether pruner prunes or not
            (If value is larger than upper, it prunes)
        n_warmup_steps:
            Pruning is disabled until the trial exceeds the given number of step.
        interval_steps:
            Interval in number of steps between the pruning checks, offset by the warmup steps.
            If no value has been reported at the time of a pruning check, that particular check
            will be postponed until a value is reported. Value must be at least 1.

    """

    def __init__(
            self,
            lower: Optional[float] = None,
            upper: Optional[float] = None,
            n_warmup_steps: int = 0,
            interval_steps: int = 1
    ) -> None:

        if lower is not None and upper is not None and lower > upper:
            raise ValueError('lower should be smaller than upper')
        if n_warmup_steps < 0:
            raise ValueError(
                'Number of warmup steps cannot be negative but got {}.'.format(n_warmup_steps))
        if interval_steps < 1:
            raise ValueError(
                'Pruning interval steps must be at least 1 but got {}.'.format(interval_steps))

        self.lower = lower
        self.upper = upper
        self._n_warmup_steps = n_warmup_steps
        self._interval_steps = interval_steps

    def prune(self, study: 'optuna.study.Study', trial: optuna.structs.FrozenTrial) -> bool:

        step = trial.last_step
        if step is None:
            return False

        n_warmup_steps = self._n_warmup_steps
        if step <= n_warmup_steps:
            return False

        if not _is_first_in_interval_step(
            step, trial.intermediate_values.keys(), n_warmup_steps,
                self._interval_steps):
            return False

        latest_value = trial.intermediate_values[step]
        if math.isnan(latest_value):
            return True

        if self.lower is not None and latest_value < self.lower:
            return True

        if self.upper is not None and latest_value > self.upper:
            return True

        return False
