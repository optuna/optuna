import numpy as np

from optuna.multi_objective.hypervolume import BaseHypervolume
from optuna.multi_objective.hypervolume import compute_2points_volume
from optuna.multi_objective.hypervolume import dom
from optuna.multi_objective.hypervolume import DomRelation
from optuna.multi_objective.hypervolume import Exact2d


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
        self._validate(solution_set, reference_point)
        self._r = reference_point
        return self._compute_rec(solution_set)

    def _compute_rec(self, solution_set: np.ndarray) -> float:
        n_points = solution_set.shape[0]
        dim_bound = solution_set.shape[1]

        if n_points == 1:
            return compute_2points_volume(solution_set[0], self._r, dim_bound)
        elif n_points == 2:
            v = 0.0
            v += compute_2points_volume(solution_set[0], self._r, dim_bound)
            v += compute_2points_volume(solution_set[1], self._r, dim_bound)
            l_edges_for_intersection = self._r - np.maximum(solution_set[0], solution_set[1])
            v -= np.prod(l_edges_for_intersection[:dim_bound])

            return v

        # n_points >= 3
        if dim_bound == 2:
            return Exact2d().compute(solution_set[:, :2], self._r[:2])

        solution_set = np.asarray(sorted(solution_set, key=lambda s: s[0], reverse=True))

        # n_points >= 3 and self._slice >= 3
        v = 0.0
        for i in range(n_points):
            v += self._compute_exclusive_hv(solution_set[i], solution_set[i + 1 :])
        return v

    def _compute_exclusive_hv(self, p: np.ndarray, s: np.ndarray) -> float:
        dim_bound = p.shape[0]
        v = compute_2points_volume(p, self._r, dim_bound)
        limited_s = self._limit(p, s)
        n_points_of_s = limited_s.shape[0]
        if n_points_of_s == 1:
            v -= compute_2points_volume(limited_s[0], self._r, dim_bound)
        elif n_points_of_s > 1:
            v -= self._compute_rec(limited_s)
        return v

    @staticmethod
    def _limit(p: np.ndarray, s: np.ndarray) -> np.ndarray:
        n_points_of_s = s.shape[0]
        dim = p.shape[0]
        limited_s = []
        p_is_not_inserted = True

        for i in range(n_points_of_s):
            if dom(p, s[i]) == DomRelation.P2_DOM_P1:
                if p_is_not_inserted:
                    assert all(p == np.maximum(s[i], p))
                    limited_s.append(np.maximum(s[i], p))
                    p_is_not_inserted = False
                else:
                    continue
            limited_s.append(np.maximum(s[i], p))

        limited_s = np.asarray(limited_s).reshape((len(limited_s), dim))
        return limited_s
