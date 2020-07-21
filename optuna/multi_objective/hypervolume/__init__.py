import enum
from typing import Optional

import numpy as np


def _validate_2_points(point1: np.ndarray, point2: np.ndarray, dim_bound: int) -> None:
    if not (point1.shape == point2.shape):
        raise ValueError("Given 2 points must have same shape.")

    if not (point1.ndim == 1 and point2.ndim == 1):
        raise ValueError("Given 2 points must be 1-d array.")

    if dim_bound <= 0:
        raise ValueError("The given dimension bound must be a positive integer.")

    if not (point1.shape[0] >= dim_bound and point2.shape[0] >= dim_bound):
        raise ValueError(
            "The length of the given 2 points must be greater than or equal to the `dim_bound`."
        )


def compute_2points_volume(
    point1: np.ndarray, point2: np.ndarray, dim_bound: Optional[int] = None
) -> float:
    """Compute the hypervolume of the hypercube, whose diagonal endpoints are given 2 points.

    Args:
        point1:
            The first endpoint of the hypercube's diagonal.
        point2:
            The second endpoint of the hypercube's diagonal.
        dim_bound:
            The bound of the dimension to compute the hypervolume.
    """

    if dim_bound is None:
        dim_bound = point1.shape[0]

    _validate_2_points(point1, point2, dim_bound)
    return float(np.abs(np.prod(point1[:dim_bound] - point2[:dim_bound])))


class DomRelation(enum.Enum):
    """Domination relationship for hypervolume computation.

    Attributes:
        P1_DOM_P2:
            The point1 dominates the point2.
        P2_DOM_P1:
            The point2 dominates the point1.
        EQUAL:
            The given 2 points are equal.
        INCOMPARABLE:
            The given 2 points are incomparable.
    """

    P1_DOM_P2 = 0
    P2_DOM_P1 = 1
    EQUAL = 2
    INCOMPARABLE = 3


def dom(point1: np.ndarray, point2: np.ndarray, dim_bound: Optional[int] = None) -> DomRelation:
    """Compare given 2 points based on domination relationship.

    Args:
        point1:
            The first point,
        point2:
            The second point.
        dim_bound:
            The bound of the dimension to compare the domination relationship.
    """
    if dim_bound is None or dim_bound == 0:
        dim_bound = point1.shape[0]

    _validate_2_points(point1, point2, dim_bound)

    for i in range(dim_bound):
        if point1[i] > point2[i]:
            if any([point1[j] < point2[j] for j in range(i + 1, dim_bound)]):
                return DomRelation.INCOMPARABLE
            else:
                return DomRelation.P2_DOM_P1
        elif point1[i] < point2[i]:
            if any([point1[j] > point2[j] for j in range(i + 1, dim_bound)]):
                return DomRelation.INCOMPARABLE
            else:
                return DomRelation.P1_DOM_P2
    return DomRelation.EQUAL


from optuna.multi_objective.hypervolume.base import BaseHypervolume  # NOQA
from optuna.multi_objective.hypervolume.exact_2d import Exact2d  # NOQA
from optuna.multi_objective.hypervolume.wfg import WFG  # NOQA
