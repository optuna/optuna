from __future__ import annotations

import math

import numpy as np

from optuna._hypervolume.wfg import compute_hypervolume


def _solve_hssp_2d(
    rank_i_loss_vals: np.ndarray,
    rank_i_indices: np.ndarray,
    subset_size: int,
    reference_point: np.ndarray,
) -> np.ndarray:
    # This function can be used for non-unique rank_i_loss_vals as well.
    # The time complexity is O(subset_size * rank_i_loss_vals.shape[0]).
    assert rank_i_loss_vals.shape[-1] == 2 and subset_size <= rank_i_loss_vals.shape[0]
    n_trials = rank_i_loss_vals.shape[0]
    # rank_i_loss_vals is unique-lexsorted in solve_hssp.
    sorted_indices = np.arange(rank_i_loss_vals.shape[0])
    sorted_loss_vals = rank_i_loss_vals.copy()
    # The diagonal points for each rectangular to calculate the hypervolume contributions.
    rect_diags = np.repeat(reference_point[np.newaxis, :], n_trials, axis=0)
    selected_indices = np.zeros(subset_size, dtype=int)
    for i in range(subset_size):
        contribs = np.prod(rect_diags - sorted_loss_vals, axis=-1)
        max_index = np.argmax(contribs)
        selected_indices[i] = rank_i_indices[sorted_indices[max_index]]
        loss_vals = sorted_loss_vals[max_index].copy()

        keep = np.ones(n_trials - i, dtype=bool)
        keep[max_index] = False
        # Remove the chosen point.
        sorted_indices = sorted_indices[keep]
        rect_diags = rect_diags[keep]
        sorted_loss_vals = sorted_loss_vals[keep]
        # Update the diagonal points for each hypervolume contribution calculation.
        rect_diags[:max_index, 0] = np.minimum(loss_vals[0], rect_diags[:max_index, 0])
        rect_diags[max_index:, 1] = np.minimum(loss_vals[1], rect_diags[max_index:, 1])

    return selected_indices


def _lazy_contribs_update(
    contribs: np.ndarray,
    pareto_loss_values: np.ndarray,
    selected_vecs: np.ndarray,
    reference_point: np.ndarray,
    hv_selected: float,
) -> np.ndarray:
    """Lazy update the hypervolume contributions.

    (1) Lazy update of the hypervolume contributions
    S=selected_indices - {indices[max_index]}, T=selected_indices, and S' is a subset of S.
    As we would like to know argmax H(T v {i}) in the next iteration, we can skip HV
    calculations for j if H(T v {i}) - H(T) > H(S' v {j}) - H(S') >= H(T v {j}) - H(T).
    We used the submodularity for the inequality above. As the upper bound of contribs[i] is
    H(S' v {j}) - H(S'), we start to update from i with a higher upper bound so that we can
    skip more HV calculations.

    (2) A simple cheap-to-evaluate contribution upper bound
    The HV difference only using the latest selected point and a candidate is a simple, yet
    obvious, contribution upper bound. Denote t as the latest selected index and j as an unselected
    index. Then, H(T v {j}) - H(T) <= H({t} v {j}) - H({t}) holds where the inequality comes from
    submodularity. We use the inclusion-exclusion principle to calculate the RHS.
    """
    if math.isinf(hv_selected):
        # NOTE(nabenabe): This part eliminates the possibility of inf - inf in this function.
        return np.full_like(contribs, np.inf)

    intersec = np.maximum(pareto_loss_values[:, np.newaxis], selected_vecs[:-1])
    inclusive_hvs = np.prod(reference_point - pareto_loss_values, axis=1)
    is_contrib_inf = np.isinf(inclusive_hvs)  # NOTE(nabe): inclusive_hvs[i] >= contribs[i].
    contribs = np.minimum(  # Please see (2) in the docstring for more details.
        contribs, inclusive_hvs - np.prod(reference_point - intersec[:, -1], axis=1)
    )
    max_contrib = 0.0
    is_hv_calc_fast = pareto_loss_values.shape[1] <= 3
    for i in np.argsort(-contribs):  # Check from larger upper bound contribs to skip more.
        if is_contrib_inf[i]:
            max_contrib = contribs[i] = np.inf
            continue
        if contribs[i] < max_contrib:  # Please see (1) in the docstring for more details.
            continue

        # NOTE(nabenabe): contribs[i] = H(S v {i)) - H(S) = H({i}) - H(S ^ {i}).
        # If HV calc is fast, the decremental approach, which involves Pareto checks, is slower.
        if is_hv_calc_fast:  # Use contribs[i] = H(S v {i)) - H(S) (incremental approach).
            selected_vecs[-1] = pareto_loss_values[i].copy()
            hv_plus = compute_hypervolume(selected_vecs, reference_point, assume_pareto=True)
            contribs[i] = hv_plus - hv_selected
        else:  # Use contribs[i] = H({i}) - H(S ^ {i}) (decremental approach).
            contribs[i] = inclusive_hvs[i] - compute_hypervolume(intersec[i], reference_point)
        max_contrib = max(contribs[i], max_contrib)

    return contribs


def _solve_hssp_on_unique_loss_vals(
    rank_i_loss_vals: np.ndarray,
    rank_i_indices: np.ndarray,
    subset_size: int,
    reference_point: np.ndarray,
) -> np.ndarray:
    if not np.isfinite(reference_point).all():
        return rank_i_indices[:subset_size]
    if rank_i_indices.size == subset_size:
        return rank_i_indices
    if rank_i_loss_vals.shape[-1] == 2:
        return _solve_hssp_2d(rank_i_loss_vals, rank_i_indices, subset_size, reference_point)

    assert subset_size < rank_i_indices.size
    # The following logic can be used for non-unique rank_i_loss_vals as well.
    diff_of_loss_vals_and_ref_point = reference_point - rank_i_loss_vals
    (n_solutions, n_objectives) = rank_i_loss_vals.shape
    contribs = np.prod(diff_of_loss_vals_and_ref_point, axis=-1)
    selected_indices = np.zeros(subset_size, dtype=int)
    selected_vecs = np.empty((subset_size, n_objectives))
    indices = np.arange(n_solutions)
    hv = 0
    for k in range(subset_size):
        max_index = int(np.argmax(contribs))
        hv += contribs[max_index]
        selected_indices[k] = indices[max_index]
        selected_vecs[k] = rank_i_loss_vals[max_index].copy()
        keep = np.ones(contribs.size, dtype=bool)
        keep[max_index] = False
        contribs = contribs[keep]
        indices = indices[keep]
        rank_i_loss_vals = rank_i_loss_vals[keep]
        if k == subset_size - 1:
            # We do not need to update contribs at the last iteration.
            break

        contribs = _lazy_contribs_update(
            contribs, rank_i_loss_vals, selected_vecs[: k + 2], reference_point, hv
        )

    return rank_i_indices[selected_indices]


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
       <https://doi.org/10.1162/EVCO_a_00188>`__
    """
    if subset_size == rank_i_indices.size:
        return rank_i_indices

    rank_i_unique_loss_vals, indices_of_unique_loss_vals = np.unique(
        rank_i_loss_vals, return_index=True, axis=0
    )
    n_unique = indices_of_unique_loss_vals.size
    if n_unique < subset_size:
        chosen = np.zeros(rank_i_indices.size, dtype=bool)
        chosen[indices_of_unique_loss_vals] = True
        duplicated_indices = np.arange(rank_i_indices.size)[~chosen]
        chosen[duplicated_indices[: subset_size - n_unique]] = True
        return rank_i_indices[chosen]

    selected_indices_of_unique_loss_vals = _solve_hssp_on_unique_loss_vals(
        rank_i_unique_loss_vals, indices_of_unique_loss_vals, subset_size, reference_point
    )
    return rank_i_indices[selected_indices_of_unique_loss_vals]
