from typing import Optional

import numpy as np


def _dominates_or_equal(
    point1: np.ndarray, point2: np.ndarray, dim_bound: Optional[int] = None
) -> bool:
    """Compare given 2 points based on domination relationship.

    Args:
        point1:
            The first point,
        point2:
            The second point.
        dim_bound:
            The bound of the dimension to compare the domination relationship.

    Returns:
        A boolean value representing whether the point1 dominates or equal to point2.
    """
    if dim_bound is None or dim_bound == 0:
        dim_bound = point1.shape[0]

    assert isinstance(dim_bound, int)

    for i in range(dim_bound):
        if point1[i] < point2[i]:
            return False  # Incompatible or point2 dominates point1
        elif point1[i] > point2[i]:
            if any([point1[j] < point2[j] for j in range(i + 1, dim_bound)]):
                return False  # Incompatible
            else:
                return True  # point1 dominates point2
    return True
