import numpy as np

from optuna.samplers._gp.optimizer.base import BaseOptimizer
from optuna.samplers._gp.optimizer.scipy import ScipyOptimizer


def optimizer_selector(optimizer: str, bounds: np.ndarray) -> BaseOptimizer:
    """Selector module for acquisition optimizers."""

    if optimizer == "L-BFGS-B":
        return ScipyOptimizer(bounds=bounds, method=optimizer)
    else:
        raise ValueError("The optimizer {} is not supported.".format(optimizer))
