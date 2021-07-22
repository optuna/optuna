from typing import Optional

import numpy as np

from optuna._hypervolume import _compute_2d
from optuna._hypervolume import _compute_2points_volume
from optuna._hypervolume import BaseHypervolume


class WFG(BaseHypervolume):
    """Hypervolume calculator for any dimension.

    This class exactly calculates the hypervolume for any dimension by using the WFG algorithm.
    For detail, see `While, Lyndon, Lucas Bradstreet, and Luigi Barone. "A fast way of
    calculating exact hypervolumes." Evolutionary Computation, IEEE Transactions on 16.1 (2012)
    : 86-95.`.
    """

    def __init__(self) -> None:
        self._reference_point: Optional[np.ndarray] = None

    def _compute(self, solution_set: np.ndarray, reference_point: np.ndarray) -> float:
        self._reference_point = reference_point.astype(np.float64)
        return self._compute_rec(solution_set.astype(np.float64))

    def _compute_rec(self, solution_set: np.ndarray) -> float:
        assert self._reference_point is not None
        n_points = solution_set.shape[0]

        if self._reference_point.shape[0] == 2:
            return _compute_2d(solution_set, self._reference_point)

        if n_points == 1:
            return _compute_2points_volume(solution_set[0], self._reference_point)
        elif n_points == 2:
            volume = 0.0
            volume += _compute_2points_volume(solution_set[0], self._reference_point)
            volume += _compute_2points_volume(solution_set[1], self._reference_point)
            intersection = self._reference_point - np.maximum(solution_set[0], solution_set[1])
            volume -= np.prod(intersection)

            return volume

        solution_set = solution_set[solution_set[:, 0].argsort()]

        # n_points >= 3
        volume = 0.0
        for i in range(n_points):
            volume += self._compute_exclusive_hv(solution_set[i], solution_set[i + 1 :])
        return volume

    def _compute_exclusive_hv(self, point: np.ndarray, solution_set: np.ndarray) -> float:
        assert self._reference_point is not None
        volume = _compute_2points_volume(point, self._reference_point)
        limited_solution_set = self._limit(point, solution_set)
        n_points_of_s = limited_solution_set.shape[0]
        if n_points_of_s == 1:
            volume -= _compute_2points_volume(limited_solution_set[0], self._reference_point)
        elif n_points_of_s > 1:
            volume -= self._compute_rec(limited_solution_set)
        return volume

    @staticmethod
    def _limit(point: np.ndarray, solution_set: np.ndarray) -> np.ndarray:
        """Limit the points in the solution set for the given point.

        Let `S := solution set`, `p := point` and `d := dim(p)`.
        The returned solution set `S'` is
        `S' = Pareto({s' | for all i in [d], exists s in S, s'_i = max(s_i, p_i)})`,
        where `Pareto(T) = the points in T which are Pareto optimal`.
        """
        n_points_of_s = solution_set.shape[0]

        limited_solution_set = np.maximum(solution_set, point)

        # Return almost Pareto optimal points for computational efficiency.
        # If the points in the solution set are completely sorted along all coordinates,
        # the following procedures return the complete Pareto optimal points.
        # For the computational efficiency, we do not completely sort the points,
        # but just sort the points according to its 0-th dimension.
        if n_points_of_s <= 1:
            return limited_solution_set
        else:
            # Assume limited_solution_set is sorted by its 0th dimension.
            # Therefore, we can simply scan the limited solution set from left to right.
            returned_limited_solution_set = [limited_solution_set[0]]
            left = 0
            right = 1
            while right < n_points_of_s:
                if (limited_solution_set[left] > limited_solution_set[right]).any():
                    left = right
                    returned_limited_solution_set.append(limited_solution_set[left])
                right += 1
            return np.asarray(returned_limited_solution_set)
