import abc

import numpy as np


class BaseHypervolume(object, metaclass=abc.ABCMeta):
    """Base class for hypervolume calculators
    """

    @abc.abstractmethod
    def compute(self, solution_set: np.ndarray, reference_point: np.ndarray) -> float:
        raise NotImplementedError
