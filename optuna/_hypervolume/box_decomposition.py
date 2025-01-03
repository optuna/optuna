from __future__ import annotations

import numpy as np
import torch

from optuna.study._multi_objective import _is_pareto_front


def _get_upper_bound_set(
    loss_vals: np.ndarray, ref_point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function follows Algorithm 2 of the paper ([Lacour17]) below:
        Title: A Box Decomposition Algorithm to Compute the Hypervolume Indicator
        Authors: Renaud Lacour, Kathrin Klamroth1, and Carlos M. Fonseca2
        URL: https://arxiv.org/abs/1510.01963

    Args:
        loss_vals: The loss values on which we build the partition.
        ref_point: The reference point.

    Returns:
        upper_bound_set: The upper bound set, which is U(N) in the paper. The shape is
        (n_bounds, n_objectives).
        def_points: The defining points of each vector in U(N). The shape is
        (n_bounds, n_objectives, n_objectives).

    NOTE:
        ``pareto_sols`` corresponds to N and ``upper_bound_set`` corresponds to U(N) in the paper.
        ``def_points`` (the shape is (n_bounds, n_objectives, n_objectives)) is not well explained
        in the paper, but basically, def_points[i, j] = z[j] of upper_bound_set[i].
    """
    (_, n_objectives) = loss_vals.shape
    objective_indices = np.arange(n_objectives)
    target_filter = ~np.eye(n_objectives, dtype=bool)
    # NOTE(nabenabe): False at 0 comes from Line 2 of Alg. 2. (loss_vals is sorted w.r.t. 0-th obj)
    target_filter[:, 0] = False

    def update(sol: np.ndarray, ubs: np.ndarray, dps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # The update rule is written in Section 2.2 of [Lacour17].
        is_dominated = np.all(sol < ubs, axis=-1)
        if not np.any(is_dominated):
            return ubs, dps

        dominated_dps = dps[is_dominated]  # The set A in Line 5 of Alg. 2.
        n_bounds = dominated_dps.shape[0]
        # NOTE(nabenabe): -inf comes from Line 2 and k!=j in Line 3 of Alg. 2.
        # NOTE(nabenabe): If True, include u in A at this index to the set in Line 3 of Alg. 2.
        include = sol >= np.max(np.where(target_filter, dominated_dps, -np.inf), axis=-2)
        # NOTE(nabenabe): The indices of u that got True in include. Each u can get True multiple
        # times for different indices j. Vectorization can process everything simultaneously.
        including_indices = np.tile(np.arange(n_bounds)[:, np.newaxis], n_objectives)[include]
        # The index `j` for each u s.t. \hat{z}_j \geq \max_{k \neq j}{z_j^k(u)}.
        target_axis = np.tile(objective_indices, (n_bounds, 1))[include]
        target_indices = np.arange(target_axis.size)
        new_dps = dominated_dps[including_indices]  # The 2nd row of the last Eq in Page 5.
        new_dps[target_indices, target_axis] = sol  # The 1st row of the same Eq.
        new_ubs = ubs[is_dominated][including_indices]  # Line 3 of Alg. 2.
        new_ubs[target_indices, target_axis] = sol[target_axis]  # \bar{z}_j in Line 3.
        return np.vstack([ubs[~is_dominated], new_ubs]), np.vstack([dps[~is_dominated], new_dps])

    upper_bound_set = np.asarray([ref_point])  # Line 1 of Alg. 2.
    def_points = np.full((1, n_objectives, n_objectives), -np.inf)  # z^k(z^r) = \hat{z}^k
    def_points[0, *np.diag_indices(n_objectives)] = ref_point  # \hat{z}^k is a dummy point.
    for sol in loss_vals[np.argsort(loss_vals[:, 0])]:
        upper_bound_set, def_points = update(sol, upper_bound_set, def_points)

    return upper_bound_set, def_points


def _get_hyper_rectangle_bounds(
    def_points: np.ndarray, upper_bound_set: np.ndarray, ref_point: np.ndarray
) -> np.ndarray:
    # Eq. (2) of [Lacour17].
    # Ref.: https://github.com/pytorch/botorch/blob/a0a2c0509dbbeec547a65f16cb0cb8d5b19fd7f1/botorch/utils/multi_objective/box_decompositions/utils.py#L164-L194
    n_objectives = upper_bound_set.shape[-1]
    bounds = np.empty((2, *upper_bound_set.shape))
    bounds[0, :, 0] = def_points[:, 0, 0]
    bounds[1, :, 0] = ref_point[0]
    row, col = np.diag_indices(n_objectives - 1)
    bounds[0, :, 1:] = np.maximum.accumulate(def_points, axis=-2)[:, row, col + 1]
    bounds[1, :, 1:] = upper_bound_set[:, 1:]
    not_empty = ~np.any(bounds[1] <= bounds[0], axis=-1)  # Remove [inf, inf] or [-inf, -inf].
    return bounds[:, not_empty]


def get_non_dominated_hyper_rectangle_bounds(
    loss_vals: np.ndarray, ref_point: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:  # (n_bounds, n_objectives) and (n_bounds, n_objectives)
    # The calculation of u[k] and l[k] in the paper below:
    # [Daulton20]: https://arxiv.org/abs/2006.05078
    # Ref.: https://github.com/pytorch/botorch/blob/a0a2c0509dbbeec547a65f16cb0cb8d5b19fd7f1/botorch/utils/multi_objective/box_decompositions/non_dominated.py#L395-L430
    # NOTE(nabenabe): The paper handles maximization problems, but we do minimization here.
    on_front = _is_pareto_front(loss_vals, assume_unique_lexsorted=False)
    upper_bound_set, _ = _get_upper_bound_set(loss_vals[on_front], ref_point)
    # Flip the sign and use upper_bound_set as the Pareto solutions. Then we can calculate the
    # lower bound set as well.
    inf_ref_point = np.full_like(ref_point, np.inf)
    neg_bound_set, neg_def_points = _get_upper_bound_set(-upper_bound_set, inf_ref_point)
    ubs, lbs = -_get_hyper_rectangle_bounds(neg_def_points, neg_bound_set, inf_ref_point)
    return torch.from_numpy(lbs), torch.from_numpy(ubs)
