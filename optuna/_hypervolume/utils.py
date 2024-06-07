import numpy as np


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
    y_cummin = np.minimum.accumulate(np.append(reference_point[1], sorted_solution_set[:-1, 1]))
    on_front = sorted_solution_set[:, 1] <= y_cummin
    pareto_solutions = sorted_solution_set[on_front]
    edge_length_x = reference_point[0] - pareto_solutions[:, 0]
    edge_length_y = y_cummin[on_front] - pareto_solutions[:, 1]
    return edge_length_x @ edge_length_y
