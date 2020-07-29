import abc

import numpy as np


class BaseHypervolume(object, metaclass=abc.ABCMeta):
    """Base class for _hypervolume calculators"""

    @abc.abstractmethod
    def compute(self, solution_set: np.ndarray, reference_point: np.ndarray) -> float:
        """Compute the _hypervolume for the given solution set and reference point.

        Args:
            solution_set:
                The solution set which we want to compute the _hypervolume.
            reference_point:
                The reference point to compute the _hypervolume.
        """

        raise NotImplementedError
