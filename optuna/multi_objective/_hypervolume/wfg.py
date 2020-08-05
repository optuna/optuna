import numpy as np

from optuna.multi_objective._hypervolume import _compute_2points_volume
from optuna.multi_objective._hypervolume import _dominates_or_equal
from optuna.multi_objective._hypervolume import BaseHypervolume


class WFG(BaseHypervolume):
    """Hypervolume calculator for any dimension.

        This class exactly calculates the hypervolume for any dimension by using the WFG algorithm.
        For detail, see `While, Lyndon, Lucas Bradstreet, and Luigi Barone. "A fast way of
        calculating exact hypervolumes." Evolutionary Computation, IEEE Transactions on 16.1 (2012)
        : 86-95.`.
    """

    def __init__(self) -> None:
        self._r = np.ndarray(shape=())

    def compute(self, solution_set: np.ndarray, reference_point: np.ndarray) -> float:
        self._r = reference_point
        return self._compute_rec(solution_set)

    def _compute_rec(self, solution_set: np.ndarray) -> float:
        n_points = solution_set.shape[0]
        dim = solution_set.shape[1]

        if n_points == 1:
            return _compute_2points_volume(solution_set[0], self._r)
        elif n_points == 2:
            v = 0.0
            v += _compute_2points_volume(solution_set[0], self._r)
            v += _compute_2points_volume(solution_set[1], self._r)
            l_edges_for_intersection = self._r - np.maximum(solution_set[0], solution_set[1])
            v -= np.prod(l_edges_for_intersection)

            return v

        solution_set = solution_set[solution_set[:, 0].argsort()]

        # n_points >= 3 and self._slice >= 3
        v = 0.0
        for i in range(n_points):
            v += self._compute_exclusive_hv(solution_set[i], solution_set[i + 1 :])
        return v

    def _compute_exclusive_hv(self, p: np.ndarray, s: np.ndarray) -> float:
        dim_bound = p.shape[0]
        v = _compute_2points_volume(p, self._r, dim_bound)
        limited_s = self._limit(p, s)
        n_points_of_s = limited_s.shape[0]
        if n_points_of_s == 1:
            v -= _compute_2points_volume(limited_s[0], self._r, dim_bound)
        elif n_points_of_s > 1:
            v -= self._compute_rec(limited_s)
        return v

    @staticmethod
    def _limit(p: np.ndarray, s: np.ndarray) -> np.ndarray:
        n_points_of_s = s.shape[0]
        dim = p.shape[0]
        limited_s = []

        for i in range(n_points_of_s):
            if _dominates_or_equal(p, s[i]):
                return p.reshape((1, dim))
            limited_s.append(np.maximum(s[i], p))
        limited_s = np.asarray(limited_s).reshape((len(limited_s), dim))

        # Return only pareto optimal points for computational efficiency.
        if n_points_of_s <= 1:
            return limited_s
        else:
            # Assume limited_s is sorted by its 0th dimension.
            returned_limited_s = [limited_s[0]]
            left = 0
            right = 1
            while left < right < n_points_of_s:
                if not _dominates_or_equal(limited_s[right], limited_s[left]):
                    left = right
                    returned_limited_s.append(limited_s[left])
                right += 1
            return np.asarray(returned_limited_s)
