from __future__ import annotations

import numpy as np
import pytest

from optuna.study import StudyDirection
from optuna.study._multi_objective import _dominates
from optuna.study._multi_objective import _fast_non_domination_rank
from optuna.study._multi_objective import _normalize_value
from optuna.trial import create_trial
from optuna.trial import TrialState


@pytest.mark.parametrize(
    ("v1", "v2"), [(-1, 1), (-float("inf"), 0), (0, float("inf")), (-float("inf"), float("inf"))]
)
def test_dominates_1d_not_equal(v1: float, v2: float) -> None:
    t1 = create_trial(values=[v1])
    t2 = create_trial(values=[v2])

    assert _dominates(t1, t2, [StudyDirection.MINIMIZE])
    assert not _dominates(t2, t1, [StudyDirection.MINIMIZE])

    assert _dominates(t2, t1, [StudyDirection.MAXIMIZE])
    assert not _dominates(t1, t2, [StudyDirection.MAXIMIZE])


@pytest.mark.parametrize("v", [0, -float("inf"), float("inf")])
@pytest.mark.parametrize("direction", [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
def test_dominates_1d_equal(v: float, direction: StudyDirection) -> None:
    assert not _dominates(create_trial(values=[v]), create_trial(values=[v]), [direction])


def test_dominates_2d() -> None:
    directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]

    # Check all pairs of trials consisting of these values, i.e.,
    # [-inf, -inf], [-inf, -1], [-inf, 1], [-inf, inf], [-1, -inf], ...
    # These values should be specified in ascending order.
    vals = [-float("inf"), -1, 1, float("inf")]

    # The following table illustrates an example of dominance relations.
    # "d" cells in the table dominates the "t" cell in (MINIMIZE, MAXIMIZE) setting.
    #
    #                        value1
    #        ╔═════╤═════╤═════╤═════╤═════╗
    #        ║     │ -∞  │ -1  │  1  │  ∞  ║
    #        ╟─────┼─────┼─────┼─────┼─────╢
    #        ║ -∞  │     │     │  d  │  d  ║
    #        ╟─────┼─────┼─────┼─────┼─────╢
    #        ║ -1  │     │     │  d  │  d  ║
    # value0 ╟─────┼─────┼─────┼─────┼─────╢
    #        ║  1  │     │     │  t  │  d  ║
    #        ╟─────┼─────┼─────┼─────┼─────╢
    #        ║  ∞  │     │     │     │     ║
    #        ╚═════╧═════╧═════╧═════╧═════╝
    #
    # In the following code, we check that for each position of "t" cell, the relation
    # above holds.

    # Generate the set of all possible indices.
    all_indices = set((i, j) for i in range(len(vals)) for j in range(len(vals)))
    for t_i, t_j in all_indices:
        # Generate the set of all indices that dominates the current index.
        dominating_indices = set(
            (d_i, d_j) for d_i in range(t_i + 1) for d_j in range(t_j, len(vals))
        )
        dominating_indices -= {(t_i, t_j)}

        for d_i, d_j in dominating_indices:
            trial1 = create_trial(values=[vals[t_i], vals[t_j]])
            trial2 = create_trial(values=[vals[d_i], vals[d_j]])
            assert _dominates(trial2, trial1, directions)

        for d_i, d_j in all_indices - dominating_indices:
            trial1 = create_trial(values=[vals[t_i], vals[t_j]])
            trial2 = create_trial(values=[vals[d_i], vals[d_j]])
            assert not _dominates(trial2, trial1, directions)


def test_dominates_invalid() -> None:
    directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]

    # The numbers of objectives for `t1` and `t2` don't match.
    t1 = create_trial(values=[1])  # One objective.
    t2 = create_trial(values=[1, 2])  # Two objectives.
    with pytest.raises(ValueError):
        _dominates(t1, t2, directions)

    # The numbers of objectives and directions don't match.
    t1 = create_trial(values=[1])  # One objective.
    t2 = create_trial(values=[1])  # One objective.
    with pytest.raises(ValueError):
        _dominates(t1, t2, directions)


@pytest.mark.parametrize("t1_state", [TrialState.FAIL, TrialState.WAITING, TrialState.PRUNED])
@pytest.mark.parametrize("t2_state", [TrialState.FAIL, TrialState.WAITING, TrialState.PRUNED])
def test_dominates_incomplete_vs_incomplete(t1_state: TrialState, t2_state: TrialState) -> None:
    directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]

    t1 = create_trial(values=None, state=t1_state)
    t2 = create_trial(values=None, state=t2_state)

    assert not _dominates(t2, t1, list(directions))
    assert not _dominates(t1, t2, list(directions))


@pytest.mark.parametrize("t1_state", [TrialState.FAIL, TrialState.WAITING, TrialState.PRUNED])
def test_dominates_complete_vs_incomplete(t1_state: TrialState) -> None:
    directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]

    t1 = create_trial(values=None, state=t1_state)
    t2 = create_trial(values=[1, 1], state=TrialState.COMPLETE)

    assert _dominates(t2, t1, list(directions))
    assert not _dominates(t1, t2, list(directions))


@pytest.mark.parametrize(
    ("trial_values", "trial_ranks"),
    [
        ([[10], [20], [20], [30]], [0, 1, 1, 2]),  # Single objective
        ([[10, 30], [10, 10], [20, 20], [30, 10], [15, 15]], [1, 0, 2, 1, 1]),  # Two objectives
        (
            [[5, 5, 4], [5, 5, 5], [9, 9, 0], [5, 7, 5], [0, 0, 9], [0, 9, 9]],
            [0, 1, 0, 2, 0, 1],
        ),  # Three objectives
        (
            [[-5, -5, -4], [-5, -5, 5], [-9, -9, 0], [5, 7, 5], [0, 0, -9], [0, -9, 9]],
            [0, 1, 0, 2, 0, 1],
        ),  # Negative values are included.
        (
            [[1, 1], [1, float("inf")], [float("inf"), 1], [float("inf"), float("inf")]],
            [0, 1, 1, 2],
        ),  # +infs are included.
        (
            [[1, 1], [1, -float("inf")], [-float("inf"), 1], [-float("inf"), -float("inf")]],
            [2, 1, 1, 0],
        ),  # -infs are included.
        (
            [[1, 1], [1, 1], [1, 2], [2, 1], [0, 1.5], [1.5, 0], [0, 1.5]],
            [0, 0, 1, 1, 0, 0, 0],
        ),  # Two objectives with duplicate values are included.
        (
            [[1, 1], [1, 1], [1, 2], [2, 1], [1, 1], [0, 1.5], [0, 1.5]],
            [0, 0, 1, 1, 0, 0, 0],
        ),  # Two objectives with duplicate values are included.
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1], [0, 1.5, 1.5], [0, 1.5, 1.5]],
            [0, 0, 1, 1, 1, 0, 0],
        ),  # Three objectives with duplicate values are included.
    ],
)
def test_fast_non_domination_rank(trial_values: list[float], trial_ranks: list[int]) -> None:
    ranks = list(_fast_non_domination_rank(np.array(trial_values)))
    assert np.array_equal(ranks, trial_ranks)


def test_fast_non_domination_rank_invalid() -> None:
    with pytest.raises(ValueError):
        _fast_non_domination_rank(
            np.array([[1.0, 2.0], [3.0, 4.0]]), penalty=np.array([1.0, 2.0, 3.0])
        )


def test_normalize_value() -> None:
    assert _normalize_value(1.0, StudyDirection.MINIMIZE) == 1.0
    assert _normalize_value(1.0, StudyDirection.MAXIMIZE) == -1.0
    assert _normalize_value(None, StudyDirection.MINIMIZE) == float("inf")
    assert _normalize_value(None, StudyDirection.MAXIMIZE) == float("inf")
