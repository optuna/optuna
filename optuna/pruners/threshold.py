from optuna.pruners import BasePruner

from optuna.type_checking import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional

    from optuna.structs import FrozenTrial
    from optuna.study import Study


class ThresholdPruner(BasePruner):
    """Pruner to detect abnormal metrics of the trials.

    Prune if a metric exceeds upper bound threshold or
    falls behind lower bound threshold which users specify.

    Example:

        .. code::

            >>> from optuna import create_study
            >>> from optuna.pruners import ThresholdPruner
            >>>
            >>> study = create_study(pruner=ThresholdPruner(upper=500))
            >>> study.optimize(objective)

    Args
        lower_bound:
            minimum value which determines whether pruner prunes or not
            (If value is smaller than lower_bound, it prunes)
        upper_bound:
            maximum value which determines whether pruner prunes or not
            (If value is larger than upper_bound, it prunes)

    """

    def __init__(self, lower_bound=None, upper_bound=None):
        # type: (Optional[float], Optional[float]) -> None

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def prune(self, study, trial):
        # type: (Study, FrozenTrial) -> bool

        last_step = trial.last_step
        if last_step is None:
            return False

        latest_value = trial.intermediate_values[last_step]

        if self.lower_bound is not None and latest_value < self.lower_bound:
            return True

        if self.upper_bound is not None and latest_value > self.upper_bound:
            return True

        return False
