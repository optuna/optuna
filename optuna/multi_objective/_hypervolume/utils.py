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
