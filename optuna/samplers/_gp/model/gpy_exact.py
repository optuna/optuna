from typing import Tuple

import numpy as np

from optuna._imports import try_import
from optuna.samplers._gp.model.base import BaseModel

with try_import() as _imports:
    import GPy


class GPyExact(BaseModel):
    """An exact Gaussian process model based on GPy library.

    .. note::
        We use GPy library. See https://github.com/SheffieldML/GPy.
    """

    def __init__(self):
        pass

    def add_data(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def predict_gradient(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass
