import operator

from optuna.pruners import BasePruner

from optuna.type_checking import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

    from optuna.structs import FrozenTrial
    from optuna.study import Study


class ThresholdPruner(BasePruner):
    """Pruner to detect abnormal metrics of the trials.

    Prune if the metric exceed a threshold.

    Example:

        .. code::

            >>> from optuna import create_study

    Args

    """

    def __init__(self, threshold, op):
        # type: (float, str) -> None

        self.threshold = threshold
        self.operator = self.get_operator_func(op)

    @staticmethod
    def get_operator_func(op):
        # type: (str) -> Callable

        # When minimize some metric (e.g. loss),
        # it prunes if the metric be extremely large.
        if op == 'gt':
            return operator.gt

        # When maximize some metric (e.g. accuracy),
        # it prunes if the metric be extremely small.
        if op == 'lt':
            return operator.lt

        # It prunes if the metric go out from a given range
        if op == 'range':
            return lambda x, y: not (-y < x < y)

        raise ValueError('Unexpected operator given')

    def prune(self, study, trial):
        # type: (Study, FrozenTrial) -> bool

        last_step = trial.last_step
        if last_step is None:
            return False

        latest_value = trial.intermediate_values[last_step]
        if self.operator(latest_value, self.threshold):
            return True

        return False
