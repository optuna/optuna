import numpy as np

from optuna.multi_objective.hypervolume import BaseHypervolume
from optuna.multi_objective.hypervolume import compute_2points_volume


class Exact2d(BaseHypervolume):
    """Hypervolume calculator in the 2-dimensional case.

    This class exactly calculates the hypervolume in the 2-dimensional case.
    """

    def __init__(self) -> None:
        pass

    def compute(self, solution_set: np.ndarray, reference_point: np.ndarray) -> float:
        self._validate(solution_set, reference_point)

        if reference_point.shape != (2,):
            raise ValueError("The dimension of given all points must be 2.")

        if solution_set.shape[0] == 1:
            return compute_2points_volume(solution_set[0], reference_point)

        sorted(solution_set, key=lambda p: p[1])

        weights = np.asarray(
            [
                reference_point[0] - np.min(solution_set[: i + 1, 0])
                for i in range(len(solution_set))
            ]
        )
        edges = np.hstack([solution_set[1:, 1], reference_point[1]]) - solution_set[:, 1]
        v = float(np.sum(weights * edges))
        return v
