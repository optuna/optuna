from typing import List

import pytest

import optuna
from optuna._multi_objective_utils import _dominates
from optuna._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def test_dominates() -> None:
    directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]

    def create_trial(values: List[float], state: TrialState = TrialState.COMPLETE) -> FrozenTrial:
        trial = optuna.trial.FrozenTrial(
            state=state,
            intermediate_values={},
            # The following attributes aren't used in this test case.
            number=0,
            value=values,
            datetime_start=None,
            datetime_complete=None,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            trial_id=0,
        )
        return trial

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
    assert _dominates(t0, t1, directions)
    assert not _dominates(t1, t0, directions)

    # `t0` dominates `t1`.
    t0 = create_trial([0, 1])
    t1 = create_trial([1, 1])
    assert _dominates(t0, t1, directions)
    assert not _dominates(t1, t0, directions)

    # `t0` and `t1` don't dominate each other.
    t0 = create_trial([1, 1])
    t1 = create_trial([1, 1])
    assert not _dominates(t0, t1, directions)
    assert not _dominates(t1, t0, directions)

    # `t0` and `t1` don't dominate each other.
    t0 = create_trial([0, 1])
    t1 = create_trial([1, 2])
    assert not _dominates(t0, t1, directions)
    assert not _dominates(t1, t0, directions)

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
            assert not _dominates(t0, t1, directions)

            if t1_state == TrialState.COMPLETE:
                # If `t0` isn't COMPLETE and `t1` is COMPLETE, `t1` dominates `t0`.
                assert _dominates(t1, t0, directions)
            else:
                # If `t1` isn't COMPLETE, it doesn't dominate others.
                assert not _dominates(t1, t0, directions)
