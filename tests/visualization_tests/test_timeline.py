from io import BytesIO
from typing import List

import pytest

import optuna
from optuna.study.study import Study
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import go
from optuna.visualization._timeline import _get_timeline_info
from optuna.visualization._timeline import plot_timeline


@pytest.mark.parametrize(
    "n",
    [0, 3],
)
def test_get_timeline_info(n: int) -> None:
    states = [TrialState.COMPLETE, TrialState.RUNNING, TrialState.WAITING]
    study = optuna.create_study()
    for i in range(n):
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": 0.5},
                distributions={"x": optuna.distributions.FloatDistribution(0, 1)},
                value=0,
                state=states[i],
            )
        )
    info = _get_timeline_info(study)
    assert len(info.bars) == n
    if n == 3:
        trials = study.get_trials()
        assert trials[0].datetime_start is not None
        assert trials[1].datetime_start is not None
        assert trials[2].datetime_start is None
        assert trials[0].datetime_complete is not None
        assert trials[1].datetime_complete is None
        assert trials[2].datetime_complete is None
        for x in info.bars:
            assert type(x.hovertext) is str
            assert x.start is not None
            assert x.end is not None
            assert x.start <= x.end


def _create_study(trial_states_list: List[TrialState]) -> Study:
    study = optuna.create_study()
    fmax = float(len(trial_states_list))
    for i, x in enumerate(trial_states_list):
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": float(i)},
                distributions={"x": optuna.distributions.FloatDistribution(-1.0, fmax)},
                value=0.0,
                state=x,
            )
        )
    return study


@pytest.mark.parametrize(
    "trial_states_list",
    [
        [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL],
        [TrialState.FAIL, TrialState.PRUNED, TrialState.COMPLETE],
    ],
)
def test_get_timeline_plot(trial_states_list: List[TrialState]) -> None:
    study = _create_study(trial_states_list)
    fig = plot_timeline(study)
    assert type(fig) is go.Figure
    fig.write_image(BytesIO())
