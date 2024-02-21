from __future__ import annotations

from io import BytesIO

import pytest

from optuna.trial import TrialState
from optuna.visualization.matplotlib._timeline import plot_timeline
from tests.visualization_tests.test_timeline import _create_study


@pytest.mark.parametrize(
    "trial_states_list",
    [
        [],
        [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL],
        [TrialState.FAIL, TrialState.PRUNED, TrialState.COMPLETE],
    ],
)
def test_get_timeline_plot(trial_states_list: list[TrialState]) -> None:
    study = _create_study(trial_states_list)
    fig = plot_timeline(study)
    fig.get_figure().savefig(BytesIO())
