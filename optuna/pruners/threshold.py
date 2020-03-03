from optuna.pruners import BasePruner
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

    """

    def __init__(self, lower=None, upper=None):
        # type: (Optional[float], Optional[float]) -> None

        self.lower = lower
        self.upper = upper

    def prune(self, study, trial):
        # type: (Study, FrozenTrial) -> bool

        last_step = trial.last_step
        if last_step is None:
            return False

        latest_value = trial.intermediate_values[last_step]

        if self.lower is not None and latest_value < self.lower:
            return True

        if self.upper is not None and latest_value > self.upper:
            return True

        return False
