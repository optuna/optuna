from typing import List
from typing import Sequence

import pytest

import optuna
from optuna.study import StudyDirection
from optuna.study._multi_objective import _dominates
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def _create_trial(values: Sequence[float], state: TrialState = TrialState.COMPLETE) -> FrozenTrial:
    return optuna.trial.create_trial(values=list(values), state=state)


@pytest.mark.parametrize(
    ("v1", "v2"), [(-1, 1), (-float("inf"), 0), (0, float("inf")), (-float("inf"), float("inf"))]
)
def test_dominates_1d_not_equal(v1: float, v2: float) -> None:
    t1 = _create_trial([v1])
    t2 = _create_trial([v2])

    assert _dominates(t1, t2, [StudyDirection.MINIMIZE])
    assert not _dominates(t2, t1, [StudyDirection.MINIMIZE])

    assert _dominates(t2, t1, [StudyDirection.MAXIMIZE])
    assert not _dominates(t1, t2, [StudyDirection.MAXIMIZE])


@pytest.mark.parametrize("v", [0, -float("inf"), float("inf")])
@pytest.mark.parametrize("direction", [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
def test_dominates_1d_equal(v: float, direction: StudyDirection) -> None:
    assert not _dominates(_create_trial([v]), _create_trial([v]), [direction])


def test_dominates_2d() -> None:
    directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]

    # Check all pairs of trials consisting of these values, i.e.,
    # [-inf, -inf], [-inf, -1], [-inf, 1], [-inf, inf], [-1, -inf], ...
    # These values should be specified in ascending order.
    vals = [-float("inf"), -1, 1, float("inf")]


    # Generate the set of all possible indices.
    all_indices = set((i, j) for i in range(len(vals)) for j in range(len(vals)))
    for (i1, j1) in all_indices:
        # Generate the set of all indices that dominates the current index.
        dominating_indices = set((i2, j2) for i2 in range(i1 + 1) for j2 in range(j1, len(vals)))
        dominating_indices -= {(i1, j1)}

        for (i2, j2) in dominating_indices:
            t1 = _create_trial([vals[i1], vals[j1]])
            t2 = _create_trial([vals[i2], vals[j2]])
            assert _dominates(t2, t1, directions)

        for (i2, j2) in all_indices - dominating_indices:
            t1 = _create_trial([vals[i1], vals[j1]])
            t2 = _create_trial([vals[i2], vals[j2]])
            assert not _dominates(t2, t1, directions)


def test_dominates_invalid() -> None:
    directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]

    # The numbers of objectives for `t1` and `t2` don't match.
    t1 = _create_trial([1])  # One objective.
    t2 = _create_trial([1, 2])  # Two objectives.
    with pytest.raises(ValueError):
        _dominates(t1, t2, directions)

    # The numbers of objectives and directions don't match.
    t1 = _create_trial([1])  # One objective.
    t2 = _create_trial([1])  # One objective.
    with pytest.raises(ValueError):
        _dominates(t1, t2, directions)


@pytest.mark.parametrize("t1_state", [TrialState.FAIL, TrialState.WAITING, TrialState.PRUNED])
@pytest.mark.parametrize("t2_state", [TrialState.FAIL, TrialState.WAITING, TrialState.PRUNED])
def test_dominates_incomplete_vs_incomplete(t1_state: TrialState, t2_state: TrialState) -> None:

    directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]

    t1 = _create_trial([1, 1], t1_state)
    t2 = _create_trial([0, 2], t2_state)

    assert not _dominates(t2, t1, list(directions))
    assert not _dominates(t1, t2, list(directions))


@pytest.mark.parametrize("t1_state", [TrialState.FAIL, TrialState.WAITING, TrialState.PRUNED])
def test_dominates_complete_vs_incomplete(t1_state: TrialState) -> None:

    directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]

    t1 = _create_trial([0, 2], t1_state)
    t2 = _create_trial([1, 1], TrialState.COMPLETE)

    assert _dominates(t2, t1, list(directions))
    assert not _dominates(t1, t2, list(directions))
