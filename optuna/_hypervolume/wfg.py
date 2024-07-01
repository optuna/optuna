from __future__ import annotations

import numpy as np

from optuna.study._multi_objective import _is_pareto_front


def _compute_2d(sorted_pareto_sols: np.ndarray, reference_point: np.ndarray) -> float:
    assert sorted_pareto_sols.shape[1] == 2 and reference_point.shape[0] == 2
    rect_diag_y = np.append(reference_point[1], sorted_pareto_sols[:-1, 1])
    edge_length_x = reference_point[0] - sorted_pareto_sols[:, 0]
    edge_length_y = rect_diag_y - sorted_pareto_sols[:, 1]
    return edge_length_x @ edge_length_y


def compute_hypervolume(
    solution_set: np.ndarray, reference_point: np.ndarray, assume_pareto: bool = False
) -> float:
    """Hypervolume calculator for any dimension.

    This class exactly calculates the hypervolume for any dimension.
    For 3 dimensions or higher, the WFG algorithm will be used.
    Please refer to ``A fast way of calculating exact hypervolumes`` for the WFG algorithm.

    .. note::
        This class is used for computing the hypervolumes of points in multi-objective space.
        Each coordinate of each point represents a ``values`` of the multi-objective function.

    .. note::
        We check that each objective is to be minimized. Transform objective values that are
        to be maximized before calling this class's ``compute`` method.

    """

    if not np.all(solution_set <= reference_point):
        raise ValueError(
            "All points must dominate or equal the reference point. "
            "That is, for all points in the solution_set and the coordinate `i`, "
            "`solution_set[i] <= reference_point[i]`."
        )
    if not np.all(np.isfinite(reference_point)):
        # reference_point does not have nan, because BaseHypervolume._validate will filter out.
        return float("inf")

    def _compute_hv(sorted_sols: np.ndarray) -> float:
        inclusive_hvs = np.prod(reference_point - sorted_sols, axis=-1)
        if inclusive_hvs.shape[0] == 1:
            return float(inclusive_hvs[0])
        elif inclusive_hvs.shape[0] == 2:
            # S(A v B) = S(A) + S(B) - S(A ^ B).
            intersec = np.prod(reference_point - np.maximum(sorted_sols[0], sorted_sols[1]))
            return np.sum(inclusive_hvs) - intersec

        limited_sols_array = np.maximum(sorted_sols[:, np.newaxis], sorted_sols)
        return sum(
            _compute_exclusive_hv(limited_sols_array[i, i + 1 :], inclusive_hv)
            for i, inclusive_hv in enumerate(inclusive_hvs)
        )

    def _compute_exclusive_hv(limited_sols: np.ndarray, inclusive_hv: float) -> float:
        if limited_sols.shape[0] == 0:
            return inclusive_hv

        on_front = _is_pareto_front(limited_sols, assume_unique_lexsorted=False)
        return inclusive_hv - _compute_hv(limited_sols[on_front])

    if not assume_pareto:
        unique_lexsorted_sols = np.unique(solution_set, axis=0)
        on_front = _is_pareto_front(unique_lexsorted_sols, assume_unique_lexsorted=True)
        sorted_pareto_sols = unique_lexsorted_sols[on_front]
    else:
        sorted_pareto_sols = solution_set[solution_set[:, 0].argsort()]

    if reference_point.shape[0] == 2:
        return _compute_2d(sorted_pareto_sols, reference_point)

    return _compute_hv(sorted_pareto_sols)
