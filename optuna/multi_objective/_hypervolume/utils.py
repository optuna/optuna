from typing import Optional

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


def _dominates_or_equal(point1: np.ndarray, point2: np.ndarray) -> bool:
    """Compare given 2 points based on domination relationship.

    Args:
        point1:
            The first point,
        point2:
            The second point.

    Returns:
        A boolean value representing whether the point1 dominates or equal to point2.
    """

    if (point1 < point2).any():
        return False
    return True
