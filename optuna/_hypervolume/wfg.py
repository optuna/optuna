from typing import Optional

import numpy as np

from optuna._hypervolume import _compute_2d
from optuna._hypervolume import BaseHypervolume
from optuna.study._multi_objective import _is_pareto_front


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
        if self._reference_point.shape[0] == 2:
            return _compute_2d(solution_set, self._reference_point)

        return self._compute_rec(solution_set[solution_set[:, 0].argsort()].astype(np.float64))

    def _compute_hv_directly(self, solution_set: np.ndarray) -> float:
        assert 1 <= solution_set.shape[0] <= 2
        if solution_set.shape[0] == 1:
            return np.abs(np.prod(solution_set[0] - self._reference_point))

        volume_sum = np.sum(np.abs(np.prod(self._reference_point - solution_set, axis=-1)))
        intersec = np.prod(self._reference_point - np.maximum(solution_set[0], solution_set[1]))
        return volume_sum - intersec

    def _compute_rec(self, solution_set: np.ndarray) -> float:
        assert self._reference_point is not None
        if solution_set.shape[0] == 1:
            return np.abs(np.prod(solution_set[0] - self._reference_point))
        elif solution_set.shape[0] == 2:
            # S(A v B) = S(A) + S(B) - S(A ^ B).
            return (
                np.sum(np.abs(np.prod(self._reference_point - solution_set, axis=-1))) -
                np.prod(self._reference_point - np.maximum(solution_set[0], solution_set[1]))
            )

        inclusive_hvs = np.abs(np.prod(self._reference_point - solution_set, axis=-1))
        limited_solutions = np.maximum(solution_set[:, np.newaxis], solution_set)
        return sum(
            self._compute_exclusive_hv(limited_solutions[i, i + 1 :], inclusive_hv)
            for i, inclusive_hv in enumerate(inclusive_hvs)
        )

    def _compute_exclusive_hv(self, limited_solution: np.ndarray, inclusive_hv: float) -> float:
        assert self._reference_point is not None
        if limited_solution.shape[0] == 0:
            return inclusive_hv
        elif limited_solution.shape[0] == 1:
            inner = np.abs(np.prod(limited_solution[0] - self._reference_point))
            return inclusive_hv - inner

        unique_lexsorted_solutions = np.unique(limited_solution, axis=0)
        on_front = _is_pareto_front(unique_lexsorted_solutions)
        limited_pareto_sols = unique_lexsorted_solutions[on_front]
        return inclusive_hv - self._compute_rec(limited_pareto_sols)
