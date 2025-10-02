from __future__ import annotations

import pytest

from optuna.trial._state import TrialState


@pytest.mark.parametrize("state", TrialState)
def test_trial_state_repr(state: TrialState) -> None:
    assert f"TrialState.{state.name}" in repr(state)
    assert str(state.value) in repr(state)
