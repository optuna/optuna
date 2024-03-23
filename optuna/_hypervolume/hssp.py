from __future__ import annotations

import numpy as np

import optuna


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
    assert subset_size != rank_i_indices.size
    assert not np.any(reference_point - rank_i_loss_vals < 0)
    n_objectives = reference_point.size
    contribs = np.prod(reference_point - rank_i_loss_vals, axis=-1)
    selected_indices = np.zeros(subset_size, dtype=int)
    selected_vecs = np.empty((subset_size, n_objectives))
    indices = np.arange(rank_i_indices.size, dtype=int)
    for k in range(subset_size):
        max_index = int(np.argmax(contribs))
        selected_indices[k] = indices[max_index]
        selected_vecs[k] = rank_i_loss_vals[indices[max_index]].copy()
        keep = np.ones(contribs.size, dtype=bool)
        keep[max_index] = False
        contribs = contribs[keep]
        indices = indices[keep]
        if k == subset_size - 1:
            break

        hv_selected = optuna._hypervolume.WFG().compute(selected_vecs[: k + 1], reference_point)
        max_contrib = 0.0
        # S = selected_indices \ {indices[max_index]}, T = selected_indices.
        # We update from contribs[i] = HV(S v {i}) - HV(S) to contribs[i] = HV(T v {i}) - HV(T).
        # However, as we skip the update time to time, contribs[i] = HV(S' v {i}) - HV(S') where
        # S' is a subset of S, so HV(S' v {i}) - HV(S') >= HV(T v {i}) - HV(T) holds
        # from submodularity. We start from i with a larger upper bound HV(S' v {i}) - HV(S').
        index_from_larger_upper_bound_contrib = np.argsort(-contribs)
        for i in index_from_larger_upper_bound_contrib:
            if contribs[i] < max_contrib:
                # Lazy evaluation to reduce HV calculations.
                # If contribs[i] will not be the maximum next, it is unnecessary to compute it.
                continue

            selected_vecs[k + 1] = rank_i_loss_vals[indices[i]].copy()
            hv_plus = optuna._hypervolume.WFG().compute(selected_vecs[: k + 2], reference_point)
            # inf - inf in the contribution calculation is always inf.
            contribs[i] = hv_plus - hv_selected if not np.isinf(hv_plus) else np.inf
            max_contrib = max(contribs[i], max_contrib)

    return rank_i_indices[selected_indices]
