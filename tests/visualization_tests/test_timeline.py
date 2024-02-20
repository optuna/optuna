from __future__ import annotations

import datetime
from io import BytesIO
from typing import Any

import _pytest.capture
import pytest

import optuna
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study.study import Study
from optuna.trial import TrialState
from optuna.visualization._plotly_imports import go
from optuna.visualization._timeline import _get_timeline_info
from optuna.visualization._timeline import plot_timeline
from tests.visualization_tests.utils import _create_study


def _create_study_negative_elapsed_time() -> Study:
    start = datetime.datetime.now()
    complete = start - datetime.timedelta(seconds=1.0)
    study = optuna.create_study()
    study.add_trial(
        optuna.trial.FrozenTrial(
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
    )
    return study


def test_get_timeline_info_empty() -> None:
    study = optuna.create_study()
    info = _get_timeline_info(study)
    assert len(info.bars) == 0


@pytest.mark.parametrize(
    "trial_sys_attrs, infeasible",
    [
        (None, False),
        ({_CONSTRAINTS_KEY: [1.0]}, True),
        ({_CONSTRAINTS_KEY: [-1.0]}, False),
    ],
)
def test_get_timeline_info(trial_sys_attrs: dict[str, Any] | None, infeasible: bool) -> None:
    states = [TrialState.COMPLETE, TrialState.RUNNING, TrialState.WAITING]
    study = _create_study(states, trial_sys_attrs)
    info = _get_timeline_info(study)
    assert len(info.bars) == len(study.get_trials())
    for bar, trial in zip(info.bars, study.get_trials()):
        assert bar.number == trial.number
        assert bar.state == trial.state
        assert type(bar.hovertext) is str
        assert isinstance(bar.start, datetime.datetime)
        assert isinstance(bar.complete, datetime.datetime)
        assert bar.start <= bar.complete
        assert bar.infeasible == infeasible


def test_get_timeline_info_negative_elapsed_time(capsys: _pytest.capture.CaptureFixture) -> None:
    # We need to reconstruct our default handler to properly capture stderr.
    optuna.logging._reset_library_root_logger()
    optuna.logging.enable_default_handler()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = _create_study_negative_elapsed_time()
    info = _get_timeline_info(study)

    _, err = capsys.readouterr()
    assert err != ""

    assert len(info.bars) == len(study.get_trials())
    for bar, trial in zip(info.bars, study.get_trials()):
        assert bar.number == trial.number
        assert bar.state == trial.state
        assert type(bar.hovertext) is str
        assert isinstance(bar.start, datetime.datetime)
        assert isinstance(bar.complete, datetime.datetime)
        assert bar.complete < bar.start


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
