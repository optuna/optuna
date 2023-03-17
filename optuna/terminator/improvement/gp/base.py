import abc
from typing import List
from typing import Tuple

import numpy as np

from optuna._experimental import experimental_class
from optuna.trial._frozen import FrozenTrial


@experimental_class("3.2.0")
class BaseGaussianProcess(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(
        self,
        trials: List[FrozenTrial],
    ) -> None:
        pass

    @abc.abstractmethod
    def predict_mean_std(
        self,
        trials: List[FrozenTrial],
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


def _min_ucb(
    trials: List[FrozenTrial],
    gp: BaseGaussianProcess,
    n_params: int,
    n_trials: int,
) -> float:
    mean, std = gp.predict_mean_std(trials)
    upper = mean + std * np.sqrt(_get_beta(n_params=n_params, n_trials=n_trials))

    return float(min(upper))


def _min_lcb(
    trials: List[FrozenTrial],
    gp: BaseGaussianProcess,
    n_params: int,
    n_trials: int,
) -> float:
    mean, std = gp.predict_mean_std(trials)
    lower = mean - std * np.sqrt(_get_beta(n_params=n_params, n_trials=n_trials))

    return float(min(lower))


def _get_beta(n_params: int, n_trials: int, delta: float = 0.1) -> float:
    beta = 2 * np.log(n_params * n_trials**2 * np.pi**2 / 6 / delta)

    # The following div is according to the original paper: "We then further scale it down
    # by a factor of 5 as defined in the experiments in Srinivas et al. (2010)"
    beta /= 5

    return beta
