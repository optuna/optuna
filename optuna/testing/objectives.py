from typing import Callable
from typing import List

from optuna import TrialPruned
from optuna.trial import Trial


def fail_objective(_: Trial) -> float:
    raise ValueError


def pruned_objective_with_intermediate_value(
    intermediate: List[float],
) -> Callable[[Trial], float]:
    def func(trial: Trial) -> float:
        for i, intermediate_value in enumerate(intermediate):
            trial.report(step=i + 1, value=intermediate_value)
        raise TrialPruned

    return func


pruned_objective = pruned_objective_with_intermediate_value([])
