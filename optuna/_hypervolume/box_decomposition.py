"""
The functions in this file are mostly based on BoTorch v0.13.0,
but they are refactored significantly from the original version.

For ``_get_upper_bound_set``, look at:
    * https://github.com/pytorch/botorch/blob/v0.13.0/botorch/utils/multi_objective/box_decompositions/utils.py#L101-L160

For ``_get_hyper_rectangle_bounds``, look at:
    * https://github.com/pytorch/botorch/blob/v0.13.0/botorch/utils/multi_objective/box_decompositions/utils.py#L163-L193

For ``_get_non_dominated_hyper_rectangle_bounds``, look at:
    * https://github.com/pytorch/botorch/blob/v0.13.0/botorch/utils/multi_objective/box_decompositions/non_dominated.py#L395-L430

The preprocessing for four or fewer objectives, we use the algorithm proposed by:
    Title: A Box Decomposition Algorithm to Compute the Hypervolume Indicator
    Authors: Renaud Lacour, Kathrin Klamroth, and Carlos M. Fonseca
    URL: https://arxiv.org/abs/1510.01963
We refer this paper as Lacour17 in this file.

"""  # NOQA: E501

from __future__ import annotations

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
        upper_bound_set: The upper bound set, which is U(N) in the paper. The shape is
        (n_bounds, n_objectives).
        def_points: The defining points of each vector in U(N). The shape is
        (n_bounds, n_objectives, n_objectives).

    NOTE:
        ``pareto_sols`` corresponds to N and ``upper_bound_set`` corresponds to U(N) in the paper.
        ``def_points`` (the shape is (n_bounds, n_objectives, n_objectives)) is not well explained
        in the paper, but basically, def_points[i, j] = z[j] of upper_bound_set[i].
    """
    (_, n_objectives) = sorted_pareto_sols.shape
    objective_indices = np.arange(n_objectives)
    target_filter = ~np.eye(n_objectives, dtype=bool)
    # NOTE(nabenabe): False at 0 comes from Line 2 of Alg. 2. (loss_vals is sorted w.r.t. 0-th obj)
    target_filter[:, 0] = False

    def update(sol: np.ndarray, ubs: np.ndarray, dps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # The update rule is written in Section 2.2 of Lacour17.
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
    def_points[0, objective_indices, objective_indices] = ref_point  # \hat{z}^k is a dummy point.
    for solution in sorted_pareto_sols:  # NOTE(nabenabe): Sorted is necessary.
        upper_bound_set, def_points = update(solution, upper_bound_set, def_points)

    return upper_bound_set, def_points


def _get_hyper_rectangle_bounds(
    def_points: np.ndarray, upper_bound_set: np.ndarray, ref_point: np.ndarray
) -> np.ndarray:
    # Eq. (2) of Lacour17.
    n_objectives = upper_bound_set.shape[-1]
    bounds = np.empty((2, *upper_bound_set.shape))
    bounds[0, :, 0] = def_points[:, 0, 0]
    bounds[1, :, 0] = ref_point[0]
    row, col = np.diag_indices(n_objectives - 1)
    bounds[0, :, 1:] = np.maximum.accumulate(def_points, axis=-2)[:, row, col + 1]
    bounds[1, :, 1:] = upper_bound_set[:, 1:]
    not_empty = ~np.any(bounds[1] <= bounds[0], axis=-1)  # Remove [inf, inf] or [-inf, -inf].
    return bounds[:, not_empty]


def _get_non_dominated_hyper_rectangle_bounds(
    sorted_pareto_sols: np.ndarray, ref_point: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:  # (n_bounds, n_objectives) and (n_bounds, n_objectives)
    # The calculation of u[k] and l[k] in the paper: https://arxiv.org/abs/2006.05078
    # NOTE(nabenabe): The paper handles maximization problems, but we do minimization here.
    _upper_bound_set = _get_upper_bound_set(sorted_pareto_sols, ref_point)[0]
    upper_bound_set = _upper_bound_set[np.argsort(_upper_bound_set[:, 0])[::-1]]
    # Flip the sign and use upper_bound_set as the Pareto solutions. Then we can calculate the
    # lower bound set as well.
    point_at_infinity = np.full_like(ref_point, np.inf)
    neg_bound_set, neg_def_points = _get_upper_bound_set(-upper_bound_set, point_at_infinity)
    ubs, lbs = -_get_hyper_rectangle_bounds(neg_def_points, neg_bound_set, point_at_infinity)
    return lbs, ubs


def _get_accepted_bound_indices(
    pareto_sols: np.ndarray, ref_point: np.ndarray, alpha: float
) -> np.ndarray:
    aug_pareto_sols = np.vstack(
        [np.min(pareto_sols, axis=0) - 1, pareto_sols, np.max(pareto_sols, axis=0) + 1]
    )
    aug_pareto_indices = np.argsort(aug_pareto_sols, axis=0)
    threshold = alpha * np.prod(aug_pareto_sols[-1] - aug_pareto_sols[0])
    (n_sols, n_objectives) = aug_pareto_sols.shape
    obj_indices = np.arange(n_objectives)
    init_bound_indices = np.tile(np.asarray([0, n_sols - 1])[:, np.newaxis], n_objectives)
    if pareto_sols.shape[0] == 0:
        return init_bound_indices[..., np.newaxis, :]

    stack = [init_bound_indices]
    accepted_bound_indices = []
    while len(stack) > 0:
        bound_indices = stack.pop()
        target_sol_indices = aug_pareto_indices[bound_indices, obj_indices]
        bs = aug_pareto_sols[target_sol_indices, obj_indices]  # Upper/Lower bounds.
        non_doms = np.all(np.any(bs[:, np.newaxis] <= pareto_sols, axis=-1), axis=-1)
        assert not isinstance(non_doms, np.bool) and non_doms.shape == (2,), "MyPy Redefinition."
        lb_non_dom, ub_non_dom = non_doms
        if ub_non_dom:  # Upper bound is non-dominated by all Pareto solutions.
            accepted_bound_indices.append(target_sol_indices)
            continue

        max_idx = np.argmax(bound_indices[1] - bound_indices[0])
        max_idx_diff = int(bound_indices[1, max_idx] - bound_indices[0, max_idx])
        if not lb_non_dom or np.prod(bs[1] - bs[0]) <= threshold or max_idx_diff <= 1:
            continue

        new_bs = np.stack([bound_indices, bound_indices])
        new_bs[*np.diag_indices(2), max_idx] += [max_idx_diff // 2, -((max_idx_diff + 1) // 2)]
        stack.extend([new_bs[1], new_bs[0]])

    return np.concat(np.asarray(accepted_bound_indices)[..., np.newaxis, :], axis=1)


def _approximate_non_dominated_hyper_rectangle_bounds(
    pareto_sols: np.ndarray, ref_point: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:  # (n_bounds, n_objectives) and (n_bounds, n_objectives)
    # See ``Towards Efficient Multiobjective Optimization: Multiobjective statistical criterions``.
    accepted_bound_indices = _get_accepted_bound_indices(pareto_sols, ref_point, alpha)
    (_, n_bounds, n_objectives) = accepted_bound_indices.shape
    aug_pareto_sols = np.vstack([np.full(n_objectives, -np.inf), pareto_sols, ref_point])
    obj_indices = np.tile(np.arange(n_objectives), 2 * n_bounds)
    return aug_pareto_sols[accepted_bound_indices.flatten(), obj_indices].reshape(
        2, n_bounds, n_objectives
    )


def get_non_dominated_hyper_rectangle_bounds(
    loss_vals: np.ndarray, ref_point: np.ndarray, alpha: float | None = None
) -> tuple[np.ndarray, np.ndarray]:  # (n_bounds, n_objectives) and (n_bounds, n_objectives)
    def _get_sorted_pareto_sols(loss_vals: np.ndarray) -> np.ndarray:
        unique_lexsorted_loss_vals = np.unique(loss_vals, axis=0)
        return unique_lexsorted_loss_vals[
            _is_pareto_front(unique_lexsorted_loss_vals, assume_unique_lexsorted=True)
        ]

    sorted_pareto_sols = _get_sorted_pareto_sols(loss_vals)
    n_objectives = loss_vals.shape[-1]
    # The condition here follows BoTorch.
    # https://github.com/pytorch/botorch/blob/v0.13.0/botorch/acquisition/multi_objective/utils.py#L55-L63
    if n_objectives <= 4:
        return _get_non_dominated_hyper_rectangle_bounds(sorted_pareto_sols, ref_point)
    else:
        alpha = alpha if alpha is not None else 10 ** (-2 if n_objectives >= 6 else -3)
        return _approximate_non_dominated_hyper_rectangle_bounds(
            sorted_pareto_sols, ref_point, alpha
        )
