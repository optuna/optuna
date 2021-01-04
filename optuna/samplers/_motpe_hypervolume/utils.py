import numpy as np


def _compute_2points_volume(point1: np.ndarray, point2: np.ndarray) -> float:
    """Compute the hypervolume of the hypercube, whose diagonal endpoints are given 2 points.

    Args:
        point1:
            The first endpoint of the hypercube's diagonal.
        point2:
            The second endpoint of the hypercube's diagonal.
    """

    return float(np.abs(np.prod(point1 - point2)))


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

    sorted_solution_set = solution_set[np.lexsort((-solution_set[:, 1], solution_set[:, 0]))]

    hypervolume = 0.0
    for solution in sorted_solution_set:
        if reference_point[1] < solution[1]:
            continue

        # Compute an area of a rectangle with reference_point
        # and one point in solution_sets as diagonals.
        hypervolume += _compute_2points_volume(reference_point, solution)

        # Update a reference_point to create a new hypervolume that
        # exclude a rectangle from the current hypervolume
        reference_point[1] = solution[1]

    return hypervolume
