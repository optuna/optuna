from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

import optuna
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_CONSTRAINTS_KEY = "constraints"


def _get_feasible_trials(trials: Sequence[FrozenTrial]) -> list[FrozenTrial]:
    feasible_trials = []
    for trial in trials:
        constraints = trial.system_attrs.get(_CONSTRAINTS_KEY)
        if constraints is None or all([x <= 0.0 for x in constraints]):
            feasible_trials.append(trial)
    return feasible_trials


def _get_pareto_front_trials_2d(
    trials: Sequence[FrozenTrial],
    directions: Sequence[StudyDirection],
    consider_constraint: bool = False,
) -> list[FrozenTrial]:
    trials = [t for t in trials if t.state == TrialState.COMPLETE]
    if consider_constraint:
        trials = _get_feasible_trials(trials)

    n_trials = len(trials)
    if n_trials == 0:
        return []

    trials.sort(
        key=lambda trial: (
            _normalize_value(trial.values[0], directions[0]),
            _normalize_value(trial.values[1], directions[1]),
        ),
    )

    last_nondominated_trial = trials[0]
    pareto_front = [last_nondominated_trial]
    for i in range(1, n_trials):
        trial = trials[i]
        if _dominates(last_nondominated_trial, trial, directions):
            continue
        pareto_front.append(trial)
        last_nondominated_trial = trial

    pareto_front.sort(key=lambda trial: trial.number)
    return pareto_front


def _get_pareto_front_trials_nd(
    trials: Sequence[FrozenTrial],
    directions: Sequence[StudyDirection],
    consider_constraint: bool = False,
) -> list[FrozenTrial]:
    pareto_front = []
    trials = [t for t in trials if t.state == TrialState.COMPLETE]
    if consider_constraint:
        trials = _get_feasible_trials(trials)

    # TODO(vincent): Optimize (use the fast non dominated sort defined in the NSGA-II paper).
    for trial in trials:
        dominated = False
        for other in trials:
            if _dominates(other, trial, directions):
                dominated = True
                break

        if not dominated:
            pareto_front.append(trial)

    return pareto_front


def _get_pareto_front_trials_by_trials(
    trials: Sequence[FrozenTrial],
    directions: Sequence[StudyDirection],
    consider_constraint: bool = False,
) -> list[FrozenTrial]:
    if len(directions) == 2:
        return _get_pareto_front_trials_2d(
            trials, directions, consider_constraint
        )  # Log-linear in number of trials.
    return _get_pareto_front_trials_nd(
        trials, directions, consider_constraint
    )  # Quadratic in number of trials.


def _get_pareto_front_trials(
    study: "optuna.study.Study", consider_constraint: bool = False
) -> list[FrozenTrial]:
    return _get_pareto_front_trials_by_trials(study.trials, study.directions, consider_constraint)


def _fast_non_dominated_sort(
    objective_values: np.ndarray,
    *,
    penalty: np.ndarray | None = None,
    n_below: int | None = None,
) -> np.ndarray:
    """Perform the fast non-dominated sort algorithm.

    The fast non-dominated sort algorithm assigns a rank to each trial based on the dominance
    relationship of the trials, determined by the objective values and the penalty values. The
    algorithm is based on `the constrained NSGA-II algorithm
    <https://doi.org/10.1109/4235.99601>`_, but the handling of the case when penalty
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
        objective_values:
            Objective values of each trials.
        penalty:
            Constraints values of each trials. Defaults to None.
        n_below: The minimum number of top trials required to be sorted. The algorithm will
            terminate when the number of sorted trials reaches n_below. Defaults to None.

    Returns:
        An ndarray in the shape of (n_trials,), where each element is the non-dominated rank of
        each trial. The rank is 0-indexed and rank -1 means that the algorithm terminated before
        the trial was sorted.
    """
    if penalty is None:
        ranks, _ = _calculate_nondomination_rank(objective_values, n_below=n_below)
        return ranks

    if len(penalty) != len(objective_values):
        raise ValueError(
            "The length of penalty and objective_values must be same, but got "
            "len(penalty)={} and len(objective_values)={}.".format(
                len(penalty), len(objective_values)
            )
        )
    nondomination_rank = np.full(len(objective_values), -1)
    is_penalty_nan = np.isnan(penalty)
    n_below = n_below or len(objective_values)

    # First, we calculate the domination rank for feasible trials.
    is_feasible = np.logical_and(~is_penalty_nan, penalty <= 0)
    ranks, bottom_rank = _calculate_nondomination_rank(
        objective_values[is_feasible], n_below=n_below
    )
    nondomination_rank[is_feasible] += 1 + ranks
    n_below -= np.count_nonzero(is_feasible)

    # Second, we calculate the domination rank for infeasible trials.
    is_infeasible = np.logical_and(~is_penalty_nan, penalty > 0)
    num_infeasible_trials = np.count_nonzero(is_infeasible)
    if num_infeasible_trials > 0:
        _, ranks = np.unique(penalty[is_infeasible], return_inverse=True)
        ranks += 1
        nondomination_rank[is_infeasible] += 1 + bottom_rank + ranks
        bottom_rank += np.max(ranks)
        n_below -= num_infeasible_trials

    # Third, we calculate the domination rank for trials with no penalty information.
    ranks, _ = _calculate_nondomination_rank(
        objective_values[is_penalty_nan], n_below=n_below, base_rank=bottom_rank + 1
    )
    nondomination_rank[is_penalty_nan] += 1 + ranks

    return nondomination_rank


def _calculate_nondomination_rank(
    objective_values: np.ndarray,
    *,
    n_below: int | None = None,
    base_rank: int = 0,
) -> tuple[np.ndarray, int]:
    if n_below is not None and n_below <= 0:
        return np.full(len(objective_values), -1), base_rank
    # Normalize n_below.
    n_below = n_below or len(objective_values)
    n_below = min(n_below, len(objective_values))

    # The ndarray `domination_mat` is a boolean 2d matrix where
    # `domination_mat[i, j] == True` means that the j-th trial dominates the i-th trial in the
    # given multi objective minimization problem.
    domination_mat = np.all(
        objective_values[:, np.newaxis, :] >= objective_values[np.newaxis, :, :], axis=2
    ) & np.any(objective_values[:, np.newaxis, :] > objective_values[np.newaxis, :, :], axis=2)

    domination_list = np.nonzero(domination_mat)
    domination_map = defaultdict(list)
    for dominated_idx, dominating_idx in zip(*domination_list):
        domination_map[dominating_idx].append(dominated_idx)

    ranks = np.full(len(objective_values), -1)
    dominated_count = np.sum(domination_mat, axis=1)

    rank = base_rank - 1
    ranked_idx_num = 0
    while ranked_idx_num < n_below:
        # Find the non-dominated trials and assign the rank.
        (non_dominated_idxs,) = np.nonzero(dominated_count == 0)
        ranked_idx_num += len(non_dominated_idxs)
        rank += 1
        ranks[non_dominated_idxs] = rank

        # Update the dominated count.
        dominated_count[non_dominated_idxs] = -1
        for non_dominated_idx in non_dominated_idxs:
            dominated_count[domination_map[non_dominated_idx]] -= 1

    return ranks, rank


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


def _normalize_value(value: None | float, direction: StudyDirection) -> float:
    if value is None:
        value = float("inf")

    if direction is StudyDirection.MAXIMIZE:
        value = -value

    return value
