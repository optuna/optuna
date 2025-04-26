"""
The functions in this file are mostly based on BoTorch v0.13.0,
but they are refactored significantly from the original version.

For ``_get_upper_bound_set``, look at:
    * https://github.com/pytorch/botorch/blob/v0.13.0/botorch/utils/multi_objective/box_decompositions/utils.py#L101-L160

For ``_get_box_bounds``, look at:
    * https://github.com/pytorch/botorch/blob/v0.13.0/botorch/utils/multi_objective/box_decompositions/utils.py#L163-L193

For ``_get_non_dominated_box_bounds``, look at:
    * https://github.com/pytorch/botorch/blob/v0.13.0/botorch/utils/multi_objective/box_decompositions/non_dominated.py#L395-L430

The preprocessing for four or fewer objectives, we use the algorithm proposed by:
    Title: A Box Decomposition Algorithm to Compute the Hypervolume Indicator
    Authors: Renaud Lacour, Kathrin Klamroth, and Carlos M. Fonseca
    URL: https://arxiv.org/abs/1510.01963
We refer this paper as Lacour17 in this file.

"""  # NOQA: E501

from __future__ import annotations

import warnings

import numpy as np

from optuna.study._multi_objective import _is_pareto_front


def _get_upper_bound_set(
    sorted_pareto_sols: np.ndarray, ref_point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function follows Algorithm 2 of Lacour17.

    Args:
        sorted_pareto_sols: Pareto solutions sorted with respect to the first objective.
        ref_point: The reference point.

    Returns:
        upper_bound_set: The upper bound set, which is ``U(N)`` in the paper. The shape is
        ``(n_bounds, n_objectives)``.
        def_points: The defining points of each vector in ``U(N)``. The shape is
        ``(n_bounds, n_objectives, n_objectives)``.

    NOTE:
        ``pareto_sols`` corresponds to ``N`` and ``upper_bound_set`` corresponds to ``U(N)`` in the
        paper.
        ``def_points`` (the shape is ``(n_bounds, n_objectives, n_objectives)``) is not well
        explained in the paper, but basically, ``def_points[i, j] = z[j]`` of
        ``upper_bound_set[i]``.
    """
    (_, n_objectives) = sorted_pareto_sols.shape
    objective_indices = np.arange(n_objectives)
    skip_ineq_judge = np.eye(n_objectives, dtype=bool)
    # NOTE(nabenabe): True at 0 comes from Line 2 of Alg. 2. (loss_vals is sorted w.r.t. 0-th obj)
    skip_ineq_judge[:, 0] = True

    def update(sol: np.ndarray, ubs: np.ndarray, dps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # The update rule is written in Section 2.2 of Lacour17.
        is_dominated = np.all(sol < ubs, axis=-1)
        if not any(is_dominated):
            return ubs, dps

        # The defining points `z(u)` for each `u in A` in Line 5 of Alg. 2.
        dominated_dps = dps[is_dominated]
        n_bounds = dominated_dps.shape[0]
        # NOTE(nabenabe): `-inf` comes from Line 2 and `k!=j` in Line 3 of Alg. 2.
        # NOTE(nabenabe): If `update[i,j]=True`, update `ubs[i,j]` to `(z_j, u_{-j})`, i.e.,
        # `np.where(objective_indices != j, ubs[i], sol[j])` in `new_ubs`. cf. Lines 2,3 of Alg. 2.
        update = sol >= np.max(np.where(skip_ineq_judge, -np.inf, dominated_dps), axis=-2)
        # NOTE(nabenabe): The indices of `u` with `True` in update. Each `u` may yield `True`
        # multiple times for different indices `j`.
        ubs_indices_to_update = np.tile(np.arange(n_bounds)[:, np.newaxis], n_objectives)[update]
        # The dimension `j` for each `u` s.t. `\hat{z}_j \geq \max_{k \neq j}{z_j^k(u)}`.
        dimensions_to_update = np.tile(objective_indices, (n_bounds, 1))[update]
        assert ubs_indices_to_update.size == dimensions_to_update.size
        indices_for_sweeping = np.arange(dimensions_to_update.size)
        # The last Eq in Page 5.
        new_dps = dominated_dps[ubs_indices_to_update]
        new_dps[indices_for_sweeping, dimensions_to_update] = sol
        # Line 3 of Alg. 2. `sol[dimensions_to_update]` is equivalent to `\bar{z}_j`.
        new_ubs = ubs[is_dominated][ubs_indices_to_update]
        new_ubs[indices_for_sweeping, dimensions_to_update] = sol[dimensions_to_update]
        return np.vstack([ubs[~is_dominated], new_ubs]), np.vstack([dps[~is_dominated], new_dps])

    upper_bound_set = np.asarray([ref_point])  # Line 1 of Alg. 2.
    def_points = np.full((1, n_objectives, n_objectives), -np.inf)  # z^k(z^r) = \hat{z}^k
    def_points[0, objective_indices, objective_indices] = ref_point  # \hat{z}^k is a dummy point.
    for solution in sorted_pareto_sols:  # NOTE(nabenabe): Sorted must be fulfilled.
        upper_bound_set, def_points = update(solution, upper_bound_set, def_points)

    return upper_bound_set, def_points


def _get_box_bounds(
    upper_bound_set: np.ndarray, def_points: np.ndarray, ref_point: np.ndarray
) -> np.ndarray:
    # Eq. (2) of Lacour17.
    n_objectives = upper_bound_set.shape[-1]
    assert n_objectives > 1, "This function is used only for multi-objective problems."
    bounds = np.empty((2, *upper_bound_set.shape))
    bounds[0, :, 0] = def_points[:, 0, 0]
    bounds[1, :, 0] = ref_point[0]
    row, col = np.diag_indices(n_objectives - 1)
    bounds[0, :, 1:] = np.maximum.accumulate(def_points, axis=-2)[:, row, col + 1]
    bounds[1, :, 1:] = upper_bound_set[:, 1:]
    not_empty = ~np.any(bounds[1] <= bounds[0], axis=-1)  # Remove [inf, inf] or [-inf, -inf].
    return bounds[:, not_empty]


def _get_non_dominated_box_bounds(
    sorted_pareto_sols: np.ndarray, ref_point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:  # (n_bounds, n_objectives) and (n_bounds, n_objectives)
    # The calculation of u[k] and l[k] in the paper: https://arxiv.org/abs/2006.05078
    # See below for the proof of this function's validity:
    # cf. https://github.com/optuna/optuna/pull/6039#issuecomment-2831926573
    # NOTE(nabenabe): The paper handles maximization problems, but we consider minimization here.
    neg_upper_bound_set = -_get_upper_bound_set(sorted_pareto_sols, ref_point)[0]
    sorted_neg_upper_bound_set = np.unique(neg_upper_bound_set, axis=0)  # lexsort by np.unique.
    # Use the sign-flipped upper_bound_set as the Pareto solutions. Then we can calculate the
    # lower bound set as well.
    point_at_infinity = np.full_like(ref_point, np.inf)
    # NOTE(nabenabe): Since our goal is to partition the non-dominated space, we only need
    # the Pareto solutions in the `neg_upper_bound_set`.
    neg_lower_bound_set, neg_def_points = _get_upper_bound_set(
        sorted_pareto_sols=sorted_neg_upper_bound_set[
            _is_pareto_front(sorted_neg_upper_bound_set, assume_unique_lexsorted=True)
        ],
        ref_point=point_at_infinity,
    )
    box_upper_bounds, box_lower_bounds = -_get_box_bounds(
        neg_lower_bound_set, neg_def_points, point_at_infinity
    )
    return box_lower_bounds, box_upper_bounds


def get_non_dominated_box_bounds(
    loss_vals: np.ndarray, ref_point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:  # (n_bounds, n_objectives) and (n_bounds, n_objectives)
    assert np.all(np.isfinite(loss_vals)), "loss_vals must be clipped before box decomposition."
    # Remove duplications and lexsort the solutions by ``np.unique``.
    unique_lexsorted_loss_vals = np.unique(loss_vals, axis=0)
    sorted_pareto_sols = unique_lexsorted_loss_vals[
        _is_pareto_front(unique_lexsorted_loss_vals, assume_unique_lexsorted=True)
    ]
    n_objectives = loss_vals.shape[-1]
    # The condition here follows BoTorch.
    # https://github.com/pytorch/botorch/blob/v0.13.0/botorch/acquisition/multi_objective/utils.py#L55-L63
    assert n_objectives > 1, "This function is used only for multi-objective problems."
    if n_objectives > 4:
        warnings.warn(
            "Box decomposition (typically used by `GPSampler`) might be significantly slow for "
            "n_objectives > 4. Please consider using another sampler instead."
        )

    return _get_non_dominated_box_bounds(sorted_pareto_sols, ref_point)
