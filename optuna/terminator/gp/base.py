import abc
from typing import List

import optuna


class BaseMinUcbLcbEstimator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(
        self,
        trials: List[optuna.trial.FrozenTrial],
    ) -> None:
        pass

    @abc.abstractmethod
    def min_ucb(self) -> float:
        pass

    @abc.abstractmethod
    def min_lcb(self) -> float:
        pass
