

from typing import List
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.study import StudyDirection
import optuna
import pytest
from optuna.study._multi_objective import _dominates

def test_dominates() -> None:

    def create_trial(values: List[float], state:TrialState = TrialState.COMPLETE) -> FrozenTrial:
        return optuna.trial.create_trial(values=values, state=state)

    directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]

    def check_domination(t0: FrozenTrial, t1: FrozenTrial) -> None:
        assert _dominates(t0, t1, directions)
        assert not _dominates(t1, t0, directions)
    
    def check_nondomination(t0: FrozenTrial, t1: FrozenTrial) -> None:
        assert not _dominates(t0, t1, directions)
        assert not _dominates(t1, t0, directions)

    # The numbers of objectives for `t0` and `t1` don't match.
    with pytest.raises(ValueError):
        t0 = create_trial([1])  # One objective.
        t1 = create_trial([1, 2])  # Two objectives.
        _dominates(t0, t1, directions)

    # The numbers of objectives and directions don't match.
    with pytest.raises(ValueError):
        t0 = create_trial([1])  # One objective.
        t1 = create_trial([1])  # One objective.
        _dominates(t0, t1, directions)

    # `t0` dominates `t1`.
    t0 = create_trial([0, 2])
    t1 = create_trial([1, 1])
    check_domination(t0, t1)

    # `t0` dominates `t1`.
    t0 = create_trial([0, 1])
    t1 = create_trial([1, 1])
    check_domination(t0, t1)

    # `t0` dominates `t1`.
    t0 = create_trial([0, 2])
    t1 = create_trial([float("inf"), 1])
    check_domination(t0, t1)

    # `t0` dominates `t1`.
    t0 = create_trial([float("inf"), 2])
    t1 = create_trial([float("inf"), 1])
    check_domination(t0, t1)

    # `t0` dominates `t1`.
    t0 = create_trial([-float("inf"), float("inf")])
    t1 = create_trial([0, 1])
    check_domination(t0, t1)

    # `t0` and `t1` don't dominate each other.
    t0 = create_trial([1, 1])
    t1 = create_trial([1, 1])
    check_nondomination(t0, t1)

    # `t0` and `t1` don't dominate each other.
    t0 = create_trial([0, 1])
    t1 = create_trial([1, 2])
    check_nondomination(t0, t1)

    # `t0` and `t1` don't dominate each other.
    t0 = create_trial([-float("inf"), 1])
    t1 = create_trial([0, 2])
    check_nondomination(t0, t1)

    # `t0` and `t1` don't dominate each other.
    t0 = create_trial([float("inf"), float("inf")])
    t1 = create_trial([float("inf"), float("inf")])
    check_nondomination(t0, t1)

    for t0_state in [TrialState.FAIL, TrialState.WAITING, TrialState.PRUNED]:
        t0 = create_trial([1, 1], t0_state)

        for t1_state in [
            TrialState.COMPLETE,
            TrialState.FAIL,
            TrialState.WAITING,
            TrialState.PRUNED,
        ]:
            # If `t0` has not the COMPLETE state, it never dominates other trials.
            t1 = create_trial([0, 2], t1_state)
            if t1_state == TrialState.COMPLETE:
                # If `t0` isn't COMPLETE and `t1` is COMPLETE, `t1` dominates `t0`.
                check_domination(t1, t0)
            else:
                # If `t1` isn't COMPLETE, it doesn't dominate others.
                check_nondomination(t0, t1)
