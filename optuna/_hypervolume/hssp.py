from typing import List

import numpy as np

import optuna


def _solve_hssp_2d(
    rank_i_loss_vals: np.ndarray,
    rank_i_indices: np.ndarray,
    subset_size: int,
    reference_point: np.ndarray,
) -> np.ndarray:
    # The time complexity is O(subset_size * rank_i_loss_vals.shape[0]).
    assert rank_i_loss_vals.shape[-1] == 2 and subset_size <= rank_i_loss_vals.shape[0]
    n_trials = rank_i_loss_vals.shape[0]
    order = np.argsort(rank_i_loss_vals[:, 0])
    sorted_loss_vals = rank_i_loss_vals[order]
    # The diagonal points for each rectangular to calculate the hypervolume contributions.
    rect_diags = np.repeat(reference_point[np.newaxis, :], n_trials, axis=0)
    subset_indices = np.zeros(subset_size, dtype=int)
    for i in range(subset_size):
        contribs = np.prod(rect_diags - sorted_loss_vals, axis=-1)
        max_index = np.argmax(contribs)
        subset_indices[i] = rank_i_indices[order[max_index]]
        loss_vals = sorted_loss_vals[max_index].copy()

        keep = np.ones(n_trials - i, dtype=bool)
        keep[max_index] = False
        # Remove the chosen point.
        order = order[keep]
        rect_diags = rect_diags[keep]
        sorted_loss_vals = sorted_loss_vals[keep]
        # Update the diagonal points for each hypervolume contribution calculation.
        rect_diags[:max_index, 0] = np.minimum(loss_vals[0], rect_diags[:max_index, 0])
        rect_diags[max_index:, 1] = np.minimum(loss_vals[1], rect_diags[max_index:, 1])

    return subset_indices


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
       <https://doi.org/10.1162/EVCO_a_00188>`_
    """
    if rank_i_loss_vals.shape[-1] == 2:
        return _solve_hssp_2d(rank_i_loss_vals, rank_i_indices, subset_size, reference_point)

    selected_vecs: List[np.ndarray] = []
    selected_indices: List[int] = []
    contributions = [
        optuna._hypervolume.WFG().compute(np.asarray([v]), reference_point)
        for v in rank_i_loss_vals
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
