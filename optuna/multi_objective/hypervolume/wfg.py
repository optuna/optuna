import functools

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

    def __init__(self):
        self._solution_set = None
        self._reference_point = None
        self._slice = None

    def compute(self, solution_set: np.ndarray, reference_point: np.ndarray) -> float:
        self._validate(solution_set, reference_point)
        self._initialize(solution_set, reference_point)
        return self._compute_rec(1)

    def _initialize(self, solution_set: np.ndarray, reference_point: np.ndarray) -> None:
        self._reference_point = reference_point
        self._frames = np.zeros((reference_point.shape[0],) + solution_set.shape)
        self._frames[0] = solution_set
        self._frame_sizes = np.zeros((reference_point.shape[0],), dtype=int)
        self._frame_sizes[0] = solution_set.shape[0]
        self._slice = reference_point.shape[0]

    def _compute_rec(self, rec_level: int) -> float:
        points = self._frames[rec_level - 1]
        n_points = self._frame_sizes[rec_level - 1]

        if n_points == 1:
            return compute_2points_volume(
                points[0], self._reference_point, self._slice
            )
        elif n_points == 2:
            v = 0.0
            v += compute_2points_volume(points[0], self._reference_point, self._slice)
            v += compute_2points_volume(points[1], self._reference_point, self._slice)
            l_edges_for_intersection = self._reference_point - np.maximum(points[0], points[1])
            v -= np.prod(l_edges_for_intersection[:self._slice])
            return v

        # n_points >= 3
        if self._slice == 2:
            return Exact2d().compute(points[:, :2], self._reference_point[:2])

        # n_points >= 3 and self._slice >= 3
        sorted(points[:n_points], key=functools.cmp_to_key(self._comp_points))

        v = 0.
        self._slice -= 1
        for i in range(n_points):
            self._limit(i + 1, i, rec_level)
            v += np.abs((points[i, self._slice] - self._reference_point[self._slice]) * self._compute_exclusive_hv(i, rec_level))
        self._slice += 1
        return v

    def _comp_points(self, point1: np.ndarray, point2: np.ndarray) -> bool:
        for i in range(self._slice-1, -1, -1):
            if point1[i] > point2[i]:
                return True
            elif point1[i] < point2[i]:
                return False
        return False

    def _limit(self, begin_index: int, index: int, rec_level: int) -> None:
        points = self._frames[rec_level - 1]
        n_points = self._frame_sizes[rec_level - 1]
        m = 0
        p = points[index]

        for i in range(begin_index, n_points):
            if i == index:
                continue

            self._frames[rec_level, m] = np.maximum(points[i], p)

            comp_results = []
            q = self._frames[rec_level, m]
            keep_q = True

            for j in range(m):
                comp_results.append(dom(q, self._frames[rec_level, j], self._slice))
                if comp_results[j] == DomRelation.P2_DOM_P1:
                    keep_q = False
                    break

            if keep_q:
                prev = 0
                nxt = 0
                while nxt < m:
                    if comp_results[nxt] != DomRelation.P1_DOM_P2 and comp_results[nxt] != DomRelation.EQUAL:
                        if prev < nxt:
                            self._frames[rec_level, prev] = self._frames[rec_level, nxt]
                        prev += 1
                    nxt += 1
                if prev < nxt:
                    self._frames[rec_level, prev] = q
                m = prev + 1

        self._frame_sizes[rec_level] = m

    def _compute_exclusive_hv(self, index: int, rec_level: int) -> float:
        v = compute_2points_volume(self._frames[rec_level - 1, index], self._reference_point, self._slice)
        if self._frame_sizes[rec_level] == 1:
            v -= compute_2points_volume(self._frames[rec_level, 0], self._reference_point, self._slice)
        else:
            v -= self._compute_rec(rec_level + 1)
        return v
