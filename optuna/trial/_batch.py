import abc
import numpy as np
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import optuna
from optuna.multi_objective.trial import MultiObjectiveTrial
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from optuna.distributions import CategoricalChoiceType


class BaseBatchTrial(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _get_trials(self) -> Union[Sequence["optuna.trial.Trial"], Sequence[MultiObjectiveTrial]]:
        raise NotImplementedError

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False
    ) -> np.ndarray:
        return np.array(
            [t.suggest_float(name, low, high, step=step, log=log) for t in self._get_trials()]
        )

    def suggest_uniform(self, name: str, low: float, high: float) -> np.ndarray:
        return np.array([t.suggest_uniform(name, low, high) for t in self._get_trials()])

    def suggest_loguniform(self, name: str, low: float, high: float) -> np.ndarray:
        return np.array([t.suggest_loguniform(name, low, high) for t in self._get_trials()])

    # TODO(Yanase): Which type do we use for return values?
    def suggest_categorical(
        self, name: str, choices: Sequence["CategoricalChoiceType"]
    ) -> List["CategoricalChoiceType"]:
        return np.array([t.suggest_categorical(name, choices) for t in self._get_trials()])

    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> np.ndarray:
        return np.array(
            [t.suggest_discrete_uniform(name, low, high, q) for t in self._get_trials()]
        )

    def suggest_int(
        self, name: str, low: int, high: int, step: int = 1, log: bool = False
    ) -> np.ndarray:
        return np.array(
            [t.suggest_int(name, low, high, step=step, log=log) for t in self._get_trials()]
        )

    def set_user_attr(self, key: str, value: Any) -> None:
        for trial in self._get_trials():
            trial.set_user_attr(key, value)

    def set_system_attr(self, key: str, value: Any) -> None:
        for trial in self._get_trials():
            trial.set_system_attr(key, value)


class BatchTrial(BaseBatchTrial):
    def __init__(self, trials: Sequence["optuna.trial.Trial"]) -> None:
        self._trials = trials

    def _get_trials(self) -> Sequence["optuna.trial.Trial"]:
        return self._trials

    def report(self, values: np.ndarray, step: int) -> None:
        for value, trial in zip(values, self._trials):
            trial.report(value, step=step)

    def should_prune(self) -> bool:
        return all((trial.should_prune() for trial in self._get_trials()))


class BatchMultiObjectiveTrial(BaseBatchTrial):
    def __init__(self, trials: Sequence[MultiObjectiveTrial]) -> None:
        self._trials = trials

    def _get_trials(self) -> Sequence[MultiObjectiveTrial]:
        return self._trials

    def report(self, values: Sequence[Sequence[float]], step: int) -> None:
        for value, trial in zip(values, self._trials):
            trial.report(value, step)

    def _report_complete_values(self, values: Sequence[np.ndarray]) -> None:
        values = np.array(values).transpose()
        for value, trial in zip(values, self._trials):
            trial._report_complete_values(value)
