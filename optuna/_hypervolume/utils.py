import numpy as np


def _compute_2d(sorted_pareto_sols: np.ndarray, reference_point: np.ndarray) -> float:
    """Compute the hypervolume for the two-dimensional space.

    This algorithm divides a hypervolume into
    smaller rectangles and sum these areas.

    Args:
        sorted_pareto_sols:
            The solution set which we want to compute the hypervolume.
            Note that this set is assumed to be sorted by the first value and Pareto optimal.
            As this set is Pareto optimal and sorted by the first order, the second value
            monotonically decreasing.
        reference_point:
            The reference point to compute the hypervolume.
    """
    assert sorted_pareto_sols.shape[1] == 2 and reference_point.shape[0] == 2
    rect_diag_y = np.append(reference_point[1], sorted_pareto_sols[:-1, 1])
    edge_length_x = reference_point[0] - sorted_pareto_sols[:, 0]
    edge_length_y = rect_diag_y - sorted_pareto_sols[:, 1]
    return edge_length_x @ edge_length_y
