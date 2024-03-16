from __future__ import annotations

import numpy as np

from optuna._hypervolume import _compute_2d
from optuna._hypervolume import BaseHypervolume
from optuna.study._multi_objective import _is_pareto_front


def _hyper_rectangular_volume(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    # NOTE: If both p0 and p1 are 1d array, this func returns np.float, which is castable to float.
    return np.abs(np.prod(p0 - p1, axis=-1))


class WFG(BaseHypervolume):
    """Hypervolume calculator for any dimension.

    This class exactly calculates the hypervolume for any dimension by using the WFG algorithm.
    For detail, see `While, Lyndon, Lucas Bradstreet, and Luigi Barone. "A fast way of
    calculating exact hypervolumes." Evolutionary Computation, IEEE Transactions on 16.1 (2012)
    : 86-95.`.
    """

    def __init__(self) -> None:
        self._reference_point: np.ndarray | None = None

    def _compute(self, solution_set: np.ndarray, reference_point: np.ndarray) -> float:
        self._reference_point = reference_point.astype(np.float64)
        if self._reference_point.shape[0] == 2:
            return _compute_2d(solution_set, self._reference_point)

        return self._compute_hv(solution_set[solution_set[:, 0].argsort()].astype(np.float64))

    def _compute_hv(self, sorted_solutions: np.ndarray) -> float:
        assert self._reference_point is not None
        if sorted_solutions.shape[0] == 1:
            return float(_hyper_rectangular_volume(sorted_solutions[0], self._reference_point))
        elif sorted_solutions.shape[0] == 2:
            # S(A v B) = S(A) + S(B) - S(A ^ B).
            intersec_node = np.maximum(sorted_solutions[0], sorted_solutions[1])
            intersec = _hyper_rectangular_volume(self._reference_point, intersec_node)
            volume_sum = np.sum(_hyper_rectangular_volume(self._reference_point, sorted_solutions))
            return volume_sum - intersec

        inclusive_hvs = _hyper_rectangular_volume(self._reference_point, sorted_solutions)
        limited_solutions_array = np.maximum(sorted_solutions[:, np.newaxis], sorted_solutions)
        return sum(
            self._compute_exclusive_hv(limited_solutions_array[i, i + 1 :], inclusive_hv)
            for i, inclusive_hv in enumerate(inclusive_hvs)
        )

    def _compute_exclusive_hv(self, limited_solutions: np.ndarray, inclusive_hv: float) -> float:
        assert self._reference_point is not None
        if limited_solutions.shape[0] == 0:
            return inclusive_hv
        elif limited_solutions.shape[0] == 1:
            inner = float(_hyper_rectangular_volume(limited_solutions[0], self._reference_point))
            return inclusive_hv - inner

        unique_lexsorted_solutions = np.unique(limited_solutions, axis=0)
        on_front = _is_pareto_front(unique_lexsorted_solutions)
        limited_pareto_sols = unique_lexsorted_solutions[on_front]
        return inclusive_hv - self._compute_hv(limited_pareto_sols)
