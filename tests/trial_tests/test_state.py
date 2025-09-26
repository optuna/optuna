from __future__ import annotations

import pytest

from optuna.trial._state import TrialState


@pytest.mark.parametrize("state", TrialState)
def test_trial_state_repr(state: TrialState) -> None:
    assert repr(state) == f"TrialState.{state.name}"
    assert eval(repr(state)) is state
