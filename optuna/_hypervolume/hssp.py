from __future__ import annotations

import numpy as np

import optuna


def _lazy_contribs_update(
    contribs: np.ndarray,
    pareto_loss_values: np.ndarray,
    selected_vecs: np.ndarray,
    reference_point: np.ndarray,
) -> np.ndarray:
    """Lazy update the hypervolume contributions.

    S=selected_indices - {indices[max_index]}, T=selected_indices, and S' is a subset of S.
    As we would like to know argmax H(T v {i}) in the next iteration, we can skip HV
    calculations for j if H(T v {i}) - H(T) > H(S' v {j}) - H(S') >= H(T v {j}) - H(T).
    We used the submodularity for the inequality above. As the upper bound of contribs[i] is
    H(S' v {j}) - H(S'), we start to update from i with a higher upper bound so that we can
    skip more HV calculations.
    """
    hv_selected = optuna._hypervolume.WFG().compute(selected_vecs[:-1], reference_point)
    max_contrib = 0.0
    index_from_larger_upper_bound_contrib = np.argsort(-contribs)
    for i in index_from_larger_upper_bound_contrib:
        if contribs[i] < max_contrib:
            # Lazy evaluation to reduce HV calculations.
            # If contribs[i] will not be the maximum next, it is unnecessary to compute it.
            continue

        selected_vecs[-1] = pareto_loss_values[i].copy()
        hv_plus = optuna._hypervolume.WFG().compute(selected_vecs, reference_point)
        # inf - inf in the contribution calculation is always inf.
        contribs[i] = hv_plus - hv_selected if not np.isinf(hv_plus) else np.inf
        max_contrib = max(contribs[i], max_contrib)

    return contribs


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
    if subset_size == rank_i_indices.size:
        return rank_i_indices

    assert not np.any(reference_point - rank_i_loss_vals <= 0)
    # Take unique to avoid a duplication error in case rank_i_loss_vals has `inf`or `-inf`.
    rank_i_unique_loss_vals, indices_of_unique_loss_vals = np.unique(
        rank_i_loss_vals, return_index=True, axis=0
    )
    n_objectives = reference_point.size
    contribs = np.prod(reference_point - rank_i_unique_loss_vals, axis=-1)
    selected_indices = np.zeros(subset_size, dtype=int)
    selected_vecs = np.empty((subset_size, n_objectives))
    indices = np.arange(rank_i_unique_loss_vals.shape[0], dtype=int)
    for k in range(subset_size):
        max_index = int(np.argmax(contribs))
        selected_indices[k] = indices[max_index]
        selected_vecs[k] = rank_i_unique_loss_vals[max_index].copy()
        keep = np.ones(contribs.size, dtype=bool)
        keep[max_index] = False
        contribs = contribs[keep]
        indices = indices[keep]
        rank_i_unique_loss_vals = rank_i_unique_loss_vals[keep]
        if k == subset_size - 1:
            # We do not need to update contribs at the last iteration.
            break

        contribs = _lazy_contribs_update(
            contribs, rank_i_unique_loss_vals, selected_vecs[: k + 2], reference_point
        )

    return rank_i_indices[indices_of_unique_loss_vals[selected_indices]]
