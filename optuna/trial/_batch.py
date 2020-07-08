from typing import Any
from typing import List
from typing import Optional
from typing import Sequence

import optuna
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from optuna.distributions import CategoricalChoiceType
# from optuna.multi_objective.trial import MultiObjectiveTrial


class BatchTrial:
    def __init__(self, trials: Sequence["optuna.trial.Trial"]):
        self._trials = trials

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False
    ) -> List[float]:
        return [t.suggest_float(name, low, high, step=step, log=log) for t in self._trials]

    def suggest_uniform(self, name: str, low: float, high: float) -> List[float]:
        return [t.suggest_uniform(name, low, high) for t in self._trials]

    def suggest_loguniform(self, name: str, low: float, high: float) -> List[float]:
        return [t.suggest_loguniform(name, low, high) for t in self._trials]

    def suggest_categorical(
        self, name: str, choices: Sequence["CategoricalChoiceType"]
    ) -> List["CategoricalChoiceType"]:
        return [t.suggest_categorical(name, choices) for t in self._trials]

    def suggest_discrete_uniform(
        self, name: str, low: float, high: float, q: float
    ) -> List[float]:
        return [t.suggest_discrete_uniform(name, low, high, q) for t in self._trials]

    def suggest_int(
        self, name: str, low: int, high: int, step: int = 1, log: bool = False
    ) -> List[int]:
        return [t.suggest_int(name, low, high, step=step, log=log) for t in self._trials]

    def report(self, values: Sequence[float], step: int) -> None:
        for value, trial in zip(values, self._trials):
            trial.report(value, step=step)

    def should_prune(self) -> bool:
        return all((trial.should_prune() for trial in self._trials))

    def set_user_attr(self, key: str, value: Any) -> None:
        for trial in self._trials:
            trial.set_user_attr(key, value)

    def set_system_attr(self, key: str, value: Any) -> None:
        for trial in self._trials:
            trial.set_system_attr(key, value)


# class BatchMultiObjectiveTrial(BatchTrial):

#     def __init__(self, trials: Sequence[MultiObjectiveTrial]):
#         super().__init__(trials)

#     def report(self, values: Sequence[Sequence[float]], step: int) -> None:
#         for value, trial in zip(values, self._trials):
#             trial.report(value, step)
