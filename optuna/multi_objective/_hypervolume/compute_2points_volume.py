from typing import Optional

import numpy as np


def _compute_2points_volume(
    point1: np.ndarray, point2: np.ndarray, dim_bound: Optional[int] = None
) -> float:
    """Compute the _hypervolume of the hypercube, whose diagonal endpoints are given 2 points.

    Args:
        point1:
            The first endpoint of the hypercube's diagonal.
        point2:
            The second endpoint of the hypercube's diagonal.
        dim_bound:
            The bound of the dimension to compute the _hypervolume.
    """

    if dim_bound is None:
        dim_bound = point1.shape[0]

    return float(np.abs(np.prod(point1[:dim_bound] - point2[:dim_bound])))
