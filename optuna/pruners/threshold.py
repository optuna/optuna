import math

from optuna.pruners import BasePruner
from optuna.pruners.percentile import _is_first_in_interval_step
from optuna import structs
from optuna.type_checking import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

    from optuna.structs import FrozenTrial
    from optuna.study import Study


class ThresholdPruner(BasePruner):
    """Pruner to detect outlying metrics of the trials.

    Prune if a metric exceeds upper bound threshold or
    falls behind lower bound threshold.

    Example:
        .. testcode::

            from optuna import create_study
            from optuna.exceptions import TrialPruned
            from optuna.pruners import ThresholdPruner


            def objective_1(trial):
                for step in range(n_trial_step):
                    trial.report(ys_for_upper[step], step)

                    if trial.should_prune():
                        raise TrialPruned()


            def objective_2(trial):
                for step in range(n_trial_step):
                    trial.report(ys_for_lower[step], step)

                    if trial.should_prune():
                        raise TrialPruned()


            ys_for_upper = [0.0, 0.1, 0.2, 0.5, 1.2]
            ys_for_lower = [100.0, 90.0, 0.1, 0.0, -1]
            n_trial_step = 5

            study = create_study(pruner=ThresholdPruner(upper=1.0))
            study.optimize(objective_1, n_trials=10)

            study = create_study(pruner=ThresholdPruner(lower=0.0))
            study.optimize(objective_2, n_trials=10)


    Args
        lower:
            minimum value which determines whether pruner prunes or not
            (If value is smaller than lower, it prunes)
        upper:
            maximum value which determines whether pruner prunes or not
            (If value is larger than upper, it prunes)
        n_startup_trials:
            Pruning is disabled until the given number of trials finish in the same study.
        n_warmup_steps:
            Pruning is disabled until the trial exceeds the given number of step.
        interval_steps:
            Interval in number of steps between the pruning checks, offset by the warmup steps.
            If no value has been reported at the time of a pruning check, that particular check
            will be postponed until a value is reported. Value must be at least 1.

    """

    def __init__(
            self,
            lower=None,
            upper=None,
            n_startup_trials=5,
            n_warmup_steps=0,
            interval_steps=1
    ):
        # type: (Optional[float], Optional[float], int, int, int) -> None

        if n_startup_trials < 0:
            raise ValueError(
                'Number of startup trials cannot be negative but got {}.'.format(n_startup_trials))
        if n_warmup_steps < 0:
            raise ValueError(
                'Number of warmup steps cannot be negative but got {}.'.format(n_warmup_steps))
        if interval_steps < 1:
            raise ValueError(
                'Pruning interval steps must be at least 1 but got {}.'.format(interval_steps))

        self.lower = lower
        self.upper = upper
        self._n_startup_trials = n_startup_trials
        self._n_warmup_steps = n_warmup_steps
        self._interval_steps = interval_steps

    def prune(self, study, trial):
        # type: (Study, FrozenTrial) -> bool

        all_trials = study.get_trials(deepcopy=False)
        n_trials = len([t for t in all_trials if t.state == structs.TrialState.COMPLETE])

        if n_trials < self._n_startup_trials:
            return False

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
