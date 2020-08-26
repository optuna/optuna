import abc

import numpy as np


class BaseHypervolume(object, metaclass=abc.ABCMeta):
    """Base class for hypervolume calculators"""

    def compute(self, solution_set: np.ndarray, reference_point: np.ndarray) -> float:
        """Compute the hypervolume for the given solution set and reference point.

        Args:
            solution_set:
                The solution set which we want to compute the hypervolume.
            reference_point:
                The reference point to compute the hypervolume.
        """

        self._validate(solution_set, reference_point)
        return self._compute(solution_set, reference_point)

    @staticmethod
    def _validate(solution_set: np.ndarray, reference_point: np.ndarray) -> None:
        # Validates that all points in the solution set dominate or equal the reference point.
        if not (solution_set <= reference_point).all():
            raise ValueError(
                "All solution must dominate or equal the reference point. "
                "That is, for all solution in the solution_set and the coordinate `i`, "
                "`solution[i] <= reference_point[i]`."
            )

    @abc.abstractmethod
    def _compute(self, solution_set: np.ndarray, reference_point: np.ndarray) -> float:
        raise NotImplementedError
