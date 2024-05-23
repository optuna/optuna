from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optuna.study._multi_objective import _is_pareto_front


if TYPE_CHECKING:
    import sortedcontainers
else:
    from optuna._imports import _LazyImport

    sortedcontainers = _LazyImport("sortedcontainers")


def _compute_3d(solution_set: np.ndarray, reference_point: np.ndarray) -> float:
    hv = 0.0
    # NOTE(nabenabe0928): The indices of Y and Z in the sorted lists are the reverse of each other.
    nondominated_Y = sortedcontainers.SortedList([-float("inf"), reference_point[1]])
    nondominated_Z = sortedcontainers.SortedList([-float("inf"), reference_point[2]])
    unique_sols = np.unique(solution_set, axis=0)
    unique_lexsorted_pareto_sols = unique_sols[_is_pareto_front(unique_sols)]
    for loss_value in unique_lexsorted_pareto_sols:
        n_nondominated = len(nondominated_Y)
        # nondominated_Y[left - 1] < y <= nondominated_Y[left]
        left = nondominated_Y.bisect_left(loss_value[1])
        # nondominated_Z[- right - 1] < z <= nondominated_Z[-right]
        right = n_nondominated - nondominated_Z.bisect_left(loss_value[2])
        assert 0 < left <= right < n_nondominated
        diagonal_point = np.asarray([nondominated_Y[right], nondominated_Z[-left]])
        inclusive_hv = np.prod(diagonal_point - loss_value[1:])
        dominated_sols = np.stack(
            [nondominated_Y[left:right], list(reversed(nondominated_Z[-right:-left]))], axis=-1
        )
        del nondominated_Y[left:right]
        del nondominated_Z[-right:-left]
        nondominated_Y.add(loss_value[1])
        nondominated_Z.add(loss_value[2])
        hv += (inclusive_hv - _compute_2d(dominated_sols, diagonal_point)) * (
            reference_point[0] - loss_value[0]
        )

    return hv


def _compute_2d(solution_set: np.ndarray, reference_point: np.ndarray) -> float:
    """Compute the hypervolume for the two-dimensional space.

    This algorithm divides a hypervolume into
    smaller rectangles and sum these areas.

    Args:
        solution_set:
            The solution set which we want to compute the hypervolume.
        reference_point:
            The reference point to compute the hypervolume.
    """
    assert solution_set.shape[1] == 2 and reference_point.shape[0] == 2
    if not np.isfinite(reference_point).all():
        return float("inf")

    # Ascending order in the 1st objective, and descending order in the 2nd objective.
    sorted_solution_set = solution_set[np.lexsort((-solution_set[:, 1], solution_set[:, 0]))]
    reference_points_y = np.append(reference_point[1], sorted_solution_set[:-1, 1])
    reference_points_y_cummin = np.minimum.accumulate(reference_points_y)
    mask = sorted_solution_set[:, 1] <= reference_points_y_cummin

    used_solution = sorted_solution_set[mask]
    edge_length_x = reference_point[0] - used_solution[:, 0]
    edge_length_y = reference_points_y_cummin[mask] - used_solution[:, 1]
    return edge_length_x @ edge_length_y
