import abc

import numpy as np


class BaseHypervolume(object, metaclass=abc.ABCMeta):
    """Base class for hypervolume calculators
    """

    @abc.abstractmethod
    def compute(self, solution_set: np.ndarray, reference_point: np.ndarray) -> float:
        """Compute the hypervolume for the given solution set and reference point.

        Args:
            solution_set:
                The solution set which we want to compute the hypervolume.
            reference_point:
                The reference point to compute the hypervolume.
        """

        raise NotImplementedError

    @staticmethod
    def _validate(solution_set: np.ndarray, reference_point: np.ndarray) -> None:
        """Verify inputs of the `compute` method.

        Args:
            solution_set:
                The solution set which we want to compute the hypervolume.
            reference_point:
                The reference point to compute the hypervolume.
        """
        if solution_set.ndim != 2:
            raise ValueError("The given solution set must be a 2-d array.")

        if reference_point.ndim != 1:
            raise ValueError("The given reference point must be a 1-d array.")

        if any([solution_set[i].ndim != solution_set[0].ndim for i in range(solution_set.ndim)]):
            raise ValueError("The dimension of each point in the solution set must be same.")

        if solution_set[0].shape != reference_point.shape:
            raise ValueError(
                "The dimension of each point in the solution set and "
                "the reference point must be same."
            )
