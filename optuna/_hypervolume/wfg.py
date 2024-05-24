from __future__ import annotations

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
        self._reference_point: np.ndarray | None = None

    def _compute(self, solution_set: np.ndarray, reference_point: np.ndarray) -> float:
        if not np.isfinite(reference_point).all():
            return float("inf")
        self._reference_point = reference_point.astype(np.float64)
        if self._reference_point.shape[0] == 2:
            return _compute_2d(solution_set, self._reference_point)

        return self._compute_hv(solution_set[solution_set[:, 0].argsort()].astype(np.float64))

    def _compute_hv(self, sorted_sols: np.ndarray) -> float:
        assert self._reference_point is not None
        inclusive_hvs = np.prod(self._reference_point - sorted_sols, axis=-1)
        if inclusive_hvs.shape[0] == 1:
            return float(inclusive_hvs[0])
        elif inclusive_hvs.shape[0] == 2:
            # S(A v B) = S(A) + S(B) - S(A ^ B).
            intersec = np.prod(self._reference_point - np.maximum(sorted_sols[0], sorted_sols[1]))
            return np.sum(inclusive_hvs) - intersec

        limited_sols_array = np.maximum(sorted_sols[:, np.newaxis], sorted_sols)
        return sum(
            self._compute_exclusive_hv(limited_sols_array[i, i + 1 :], inclusive_hv)
            for i, inclusive_hv in enumerate(inclusive_hvs)
        )

    def _compute_exclusive_hv(self, limited_sols: np.ndarray, inclusive_hv: float) -> float:
        if limited_sols.shape[0] == 0:
            return inclusive_hv

        on_front = _is_pareto_front(limited_sols, assume_unique_lexsorted=False)
        return inclusive_hv - self._compute_hv(limited_sols[on_front])
