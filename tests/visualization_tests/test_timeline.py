from __future__ import annotations

import datetime
from io import BytesIO

import pytest

import optuna
from optuna.study.study import Study
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import go
from optuna.visualization._timeline import _get_timeline_info
from optuna.visualization._timeline import plot_timeline


def _create_study(trial_states_list: list[TrialState]) -> Study:
    study = optuna.create_study()
    fmax = float(len(trial_states_list))
    for i, s in enumerate(trial_states_list):
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": float(i)},
                distributions={"x": optuna.distributions.FloatDistribution(-1.0, fmax)},
                value=0.0,
                state=s,
            )
        )
    return study


def test_get_timeline_info_empty() -> None:
    study = optuna.create_study()
    info = _get_timeline_info(study)
    assert len(info.bars) == 0


def test_get_timeline_info() -> None:
    states = [TrialState.COMPLETE, TrialState.RUNNING, TrialState.WAITING]
    study = _create_study(states)
    info = _get_timeline_info(study)
    assert len(info.bars) == 3
    for bar, trial in zip(info.bars, study.get_trials()):
        assert bar.number == trial.number
        assert bar.state == trial.state
        assert type(bar.hovertext) is str
        assert isinstance(bar.start, datetime.datetime)
        assert isinstance(bar.end, datetime.datetime)
        assert bar.start <= bar.end


def test_get_timeline_info_reverse() -> None:
    start = datetime.datetime.now()
    complete = start - datetime.timedelta(seconds=1.0)
    frozentrial = optuna.trial.FrozenTrial(
        number=-1,
        trial_id=-1,
        state=TrialState.COMPLETE,
        value=0.0,
        values=None,
        datetime_start=start,
        datetime_complete=complete,
        params={},
        distributions={},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )
    study = optuna.create_study()
    study.add_trial(frozentrial)
    info = _get_timeline_info(study)
    assert len(info.bars) == 1
    for bar, trial in zip(info.bars, study.get_trials()):
        assert bar.number == trial.number
        assert bar.state == trial.state
        assert type(bar.hovertext) is str
        assert isinstance(bar.start, datetime.datetime)
        assert isinstance(bar.end, datetime.datetime)
        # assert bar.start <= bar.end
    fig = plot_timeline(study)
    assert type(fig) is go.Figure
    fig.write_image(BytesIO())


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
    assert type(fig) is go.Figure
    fig.write_image(BytesIO())
