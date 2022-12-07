import numpy as np
import optuna 

# from optuna._hypervolume import WFG

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


def _solve_hssp(
    rank_i_loss_vals: np.ndarray,
    rank_i_indices: np.ndarray,
    subset_size: int,
    reference_point: np.ndarray,
) -> np.ndarray:
    """Solve a hypervolume subset selection problem (HSSP) via a greedy algorithm.

    This method is a 1-1/e approximation algorithm to solve HSSP.

    For further information about algorithms to solve HSSP, please refer to the following
    paper:

    - `Greedy Hypervolume Subset Selection in Low Dimensions
       <https://ieeexplore.ieee.org/document/7570501>`_
    """
    selected_vecs = []  # type: List[np.ndarray]
    selected_indices = []  # type: List[int]
    contributions = [
        optuna._hypervolume.WFG().compute(np.asarray([v]), reference_point) for v in rank_i_loss_vals
    ]
    hv_selected = 0.0
    while len(selected_indices) < subset_size:
        max_index = int(np.argmax(contributions))
        contributions[max_index] = -1  # mark as selected
        selected_index = rank_i_indices[max_index]
        selected_vec = rank_i_loss_vals[max_index]
        for j, v in enumerate(rank_i_loss_vals):
            if contributions[j] == -1:
                continue
            p = np.max([selected_vec, v], axis=0)
            contributions[j] -= (
                optuna._hypervolume.WFG().compute(np.asarray(selected_vecs + [p]), reference_point)
                - hv_selected
            )
        selected_vecs += [selected_vec]
        selected_indices += [selected_index]
        hv_selected = optuna._hypervolume.WFG().compute(np.asarray(selected_vecs), reference_point)

    return np.asarray(selected_indices, dtype=int)
