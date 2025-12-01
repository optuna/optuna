from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import optuna
from optuna.study._constrained_optimization import _get_feasible_trials
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def _get_pareto_front_trials_by_trials(
    trials: Sequence[FrozenTrial],
    directions: Sequence[StudyDirection],
    consider_constraint: bool = False,
) -> list[FrozenTrial]:
    # NOTE(nabenabe0928): Vectorization relies on all the trials being complete.
    trials = [t for t in trials if t.state == TrialState.COMPLETE]
    if consider_constraint:
        trials = _get_feasible_trials(trials)
    if len(trials) == 0:
        return []

    if any(len(t.values) != len(directions) for t in trials):
        raise ValueError(
            "The number of the values and the number of the objectives must be identical."
        )

    loss_values = np.asarray(
        [[_normalize_value(v, d) for v, d in zip(t.values, directions)] for t in trials]
    )
    on_front = _is_pareto_front(loss_values, assume_unique_lexsorted=False)
    return [t for t, is_pareto in zip(trials, on_front) if is_pareto]


def _get_pareto_front_trials(
    study: "optuna.study.Study", consider_constraint: bool = False
) -> list[FrozenTrial]:
    return _get_pareto_front_trials_by_trials(study.trials, study.directions, consider_constraint)


def _fast_non_domination_rank(
    loss_values: np.ndarray, *, penalty: np.ndarray | None = None, n_below: int | None = None
) -> np.ndarray:
    """Calculate non-domination rank based on the fast non-dominated sort algorithm.

    The fast non-dominated sort algorithm assigns a rank to each trial based on the dominance
    relationship of the trials, determined by the objective values and the penalty values. The
    algorithm is based on `the constrained NSGA-II algorithm
    <https://doi.org/10.1109/4235.99601>`__, but the handling of the case when penalty
    values are None is different. The algorithm assigns the rank according to the following
    rules:

    1. Feasible trials: First, the algorithm assigns the rank to feasible trials, whose penalty
        values are less than or equal to 0, according to unconstrained version of fast non-
        dominated sort.
    2. Infeasible trials: Next, the algorithm assigns the rank from the minimum penalty value of to
        the maximum penalty value.
    3. Trials with no penalty information (constraints value is None): Finally, The algorithm
        assigns the rank to trials with no penalty information according to unconstrained version
        of fast non-dominated sort. Note that only this step is different from the original
        constrained NSGA-II algorithm.
    Plus, the algorithm terminates whenever the number of sorted trials reaches n_below.

    Args:
        loss_values:
            Objective values, which is better when it is lower, of each trials.
        penalty:
            Constraints values of each trials. Defaults to None.
        n_below: The minimum number of top trials required to be sorted. The algorithm will
            terminate when the number of sorted trials reaches n_below. Defaults to None.

    Returns:
        An ndarray in the shape of (n_trials,), where each element is the non-domination rank of
        each trial. The rank is 0-indexed. This function guarantees the correctness of the ranks
        only up to the top-``n_below`` solutions. If a solution's rank is worse than the
        top-``n_below`` solution, its rank will be guaranteed to be greater than the rank of
        the top-``n_below`` solution.
    """
    if len(loss_values) == 0:
        return np.array([], dtype=int)

    n_below = n_below or len(loss_values)
    assert n_below > 0, "n_below must be a positive integer."

    if penalty is None:
        return _calculate_nondomination_rank(loss_values, n_below=n_below)

    if len(penalty) != len(loss_values):
        raise ValueError(
            "The length of penalty and loss_values must be same, but got "
            f"{len(penalty)=} and {len(loss_values)=}."
        )

    ranks = np.full(len(loss_values), -1, dtype=int)
    is_penalty_nan = np.isnan(penalty)
    is_feasible = np.logical_and(~is_penalty_nan, penalty <= 0)
    is_infeasible = np.logical_and(~is_penalty_nan, penalty > 0)

    # First, we calculate the domination rank for feasible trials.
    ranks[is_feasible] = _calculate_nondomination_rank(loss_values[is_feasible], n_below=n_below)
    n_below -= int(np.count_nonzero(is_feasible))

    # Second, we calculate the domination rank for infeasible trials.
    top_rank_infeasible = np.max(ranks[is_feasible], initial=-1) + 1
    ranks[is_infeasible] = top_rank_infeasible + _calculate_nondomination_rank(
        penalty[is_infeasible][:, np.newaxis], n_below=n_below
    )
    n_below -= int(np.count_nonzero(is_infeasible))

    # Third, we calculate the domination rank for trials with no penalty information.
    top_rank_penalty_nan = np.max(ranks[~is_penalty_nan], initial=-1) + 1
    ranks[is_penalty_nan] = top_rank_penalty_nan + _calculate_nondomination_rank(
        loss_values[is_penalty_nan], n_below=n_below
    )
    assert np.all(ranks != -1), "All the rank must be updated."
    return ranks


def _is_pareto_front_nd(unique_lexsorted_loss_values: np.ndarray) -> np.ndarray:
    n_trials, n_objectives = unique_lexsorted_loss_values.shape
    is_dominated = np.zeros(n_trials, dtype=bool)

    def _dominance_check(candidates: np.ndarray, targets: np.ndarray, dim: int) -> None:
        if len(candidates) == 0 or len(targets) == 0:
            return

        if dim == n_objectives - 1:
            # 1D case: check if min(candidates) <= target
            min_val = np.min(unique_lexsorted_loss_values[candidates, dim])
            is_dominated[targets[unique_lexsorted_loss_values[targets, dim] >= min_val]] = True
            return

        if dim == n_objectives - 2:
            # 2D case: Sweep line algorithm
            # Sort candidates and targets by current dimension
            c_vals = unique_lexsorted_loss_values[candidates, dim]
            t_vals = unique_lexsorted_loss_values[targets, dim]

            c_order = np.argsort(c_vals)
            # Note: targets don't strictly need sorting for searchsorted,
            # but we need to process them. We use the original targets order for the query.

            sorted_c = candidates[c_order]
            vals_c_curr = c_vals[c_order]
            vals_c_next = unique_lexsorted_loss_values[sorted_c, dim + 1]

            cummin_c_next = np.minimum.accumulate(vals_c_next)

            # Find the rightmost candidate such that candidate.dim <= target.dim
            indices = np.searchsorted(vals_c_curr, t_vals, side="right") - 1
            valid_mask = indices >= 0

            if not np.any(valid_mask):
                return

            target_indices_to_update = targets[valid_mask]
            relevant_min_vals = cummin_c_next[indices[valid_mask]]
            relevant_t_next = unique_lexsorted_loss_values[target_indices_to_update, dim + 1]

            is_dominated[target_indices_to_update[relevant_t_next >= relevant_min_vals]] = True
            return

        # General case recursion
        all_indices = np.concatenate([candidates, targets])
        is_candidate = np.zeros(len(all_indices), dtype=bool)
        is_candidate[: len(candidates)] = True

        vals = unique_lexsorted_loss_values[all_indices, dim]
        order = np.argsort(vals)
        sorted_indices = all_indices[order]
        sorted_is_candidate = is_candidate[order]

        mid = len(all_indices) // 2

        L_indices = sorted_indices[:mid]
        L_is_cand = sorted_is_candidate[:mid]

        R_indices = sorted_indices[mid:]
        R_is_cand = sorted_is_candidate[mid:]

        L_c = L_indices[L_is_cand]
        L_t = L_indices[~L_is_cand]

        R_c = R_indices[R_is_cand]
        R_t = R_indices[~R_is_cand]

        if len(L_c) > 0 and len(L_t) > 0:
            _dominance_check(L_c, L_t, dim)

        if len(R_c) > 0 and len(R_t) > 0:
            _dominance_check(R_c, R_t, dim)

        if len(L_c) > 0 and len(R_t) > 0:
            _dominance_check(L_c, R_t, dim + 1)

    def solve(indices: np.ndarray, dim: int) -> None:
        if len(indices) <= 1:
            return

        mid = len(indices) // 2
        left = indices[:mid]
        right = indices[mid:]

        solve(left, dim)
        solve(right, dim)

        candidates = left[~is_dominated[left]]
        targets = right[~is_dominated[right]]

        if len(candidates) > 0 and len(targets) > 0:
            _dominance_check(candidates, targets, dim + 1)

    solve(np.arange(n_trials), 0)
    return ~is_dominated


def _is_pareto_front_2d(unique_lexsorted_loss_values: np.ndarray) -> np.ndarray:
    n_trials = unique_lexsorted_loss_values.shape[0]
    cummin_value1 = np.minimum.accumulate(unique_lexsorted_loss_values[:, 1])
    on_front = np.ones(n_trials, dtype=bool)
    on_front[1:] = cummin_value1[1:] < cummin_value1[:-1]  # True if cummin value1 is new minimum.
    return on_front


def _is_pareto_front_for_unique_sorted(unique_lexsorted_loss_values: np.ndarray) -> np.ndarray:
    (n_trials, n_objectives) = unique_lexsorted_loss_values.shape
    if n_objectives == 1:
        on_front = np.zeros(len(unique_lexsorted_loss_values), dtype=bool)
        on_front[0] = True  # Only the first element is Pareto optimal.
        return on_front
    elif n_objectives == 2:
        return _is_pareto_front_2d(unique_lexsorted_loss_values)
    else:
        return _is_pareto_front_nd(unique_lexsorted_loss_values)


def _is_pareto_front(loss_values: np.ndarray, assume_unique_lexsorted: bool) -> np.ndarray:
    # NOTE(nabenabe): If assume_unique_lexsorted=True, but loss_values is not a unique array,
    # Duplicated Pareto solutions will be filtered out except for the earliest occurrences.
    # If assume_unique_lexsorted=True and loss_values[:, 0] is not sorted, then the result will be
    # incorrect.
    if assume_unique_lexsorted:
        return _is_pareto_front_for_unique_sorted(loss_values)

    unique_lexsorted_loss_values, order_inv = np.unique(loss_values, axis=0, return_inverse=True)
    on_front = _is_pareto_front_for_unique_sorted(unique_lexsorted_loss_values)
    # NOTE(nabenabe): We can remove `.reshape(-1)` if ``numpy==2.0.0`` is not used.
    # https://github.com/numpy/numpy/issues/26738
    # TODO: Remove `.reshape(-1)` once `numpy==2.0.0` is obsolete.
    return on_front[order_inv.reshape(-1)]


def _calculate_nondomination_rank(
    loss_values: np.ndarray, *, n_below: int | None = None
) -> np.ndarray:
    if len(loss_values) == 0 or (n_below is not None and n_below <= 0):
        return np.zeros(len(loss_values), dtype=int)

    (n_trials, n_objectives) = loss_values.shape
    if n_objectives == 1:
        _, ranks = np.unique(loss_values[:, 0], return_inverse=True)
        return ranks

    # It ensures that trials[j] will not dominate trials[i] for i < j.
    # np.unique does lexsort.
    unique_lexsorted_loss_values, order_inv = np.unique(loss_values, return_inverse=True, axis=0)
    n_unique = unique_lexsorted_loss_values.shape[0]
    # Clip n_below.
    n_below = min(n_below or len(unique_lexsorted_loss_values), len(unique_lexsorted_loss_values))
    ranks = np.zeros(n_unique, dtype=int)
    rank = 0
    indices = np.arange(n_unique)
    while n_unique - indices.size < n_below:
        on_front = _is_pareto_front(unique_lexsorted_loss_values, assume_unique_lexsorted=True)
        ranks[indices[on_front]] = rank
        # Remove the recent Pareto solutions.
        indices = indices[~on_front]
        unique_lexsorted_loss_values = unique_lexsorted_loss_values[~on_front]
        rank += 1

    ranks[indices] = rank  # Rank worse than the top n_below is defined as the worst rank.
    # NOTE(nabenabe): We can remove `.reshape(-1)` if ``numpy==2.0.0`` is not used.
    # https://github.com/numpy/numpy/issues/26738
    # TODO: Remove `.reshape(-1)` once `numpy==2.0.0` is obsolete.
    return ranks[order_inv.reshape(-1)]


def _dominates(
    trial0: FrozenTrial, trial1: FrozenTrial, directions: Sequence[StudyDirection]
) -> bool:
    values0 = trial0.values
    values1 = trial1.values

    if trial0.state != TrialState.COMPLETE:
        return False

    if trial1.state != TrialState.COMPLETE:
        return True

    assert values0 is not None
    assert values1 is not None

    if len(values0) != len(values1):
        raise ValueError("Trials with different numbers of objectives cannot be compared.")

    if len(values0) != len(directions):
        raise ValueError(
            "The number of the values and the number of the objectives are mismatched."
        )

    normalized_values0 = [_normalize_value(v, d) for v, d in zip(values0, directions)]
    normalized_values1 = [_normalize_value(v, d) for v, d in zip(values1, directions)]

    if normalized_values0 == normalized_values1:
        return False

    return all(v0 <= v1 for v0, v1 in zip(normalized_values0, normalized_values1))


def _normalize_value(value: float | None, direction: StudyDirection) -> float:
    if value is None:
        return float("inf")

    if direction is StudyDirection.MAXIMIZE:
        value = -value

    return value
