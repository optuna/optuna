import abc
from typing import List
from typing import Tuple

import numpy as np

import optuna


class BaseGaussianProcess(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(
        self,
        trials: List[optuna.trial.FrozenTrial],
    ) -> None:
        pass

    @abc.abstractmethod
    def mean_std(
        self,
        trials: List[optuna.trial.FrozenTrial],
    ) -> Tuple[List[float], List[float]]:
        pass

    @abc.abstractmethod
    def min_ucb(self) -> float:
        pass

    @abc.abstractmethod
    def min_lcb(self, n_additional_candidates: int = 2000) -> float:
        pass

    def beta(self, delta: float = 0.1) -> float:
        beta = 2 * np.log(self.gamma() * self.t() ** 2 * np.pi**2 / 6 / delta)

        # The following div is according to the original paper: "We then further scale it down
        # by a factor of 5 as defined in the experiments in Srinivas et al. (2010)"
        beta /= 5

        return beta

    @abc.abstractmethod
    def gamma(self) -> float:
        pass

    @abc.abstractmethod
    def t(self) -> float:
        pass
