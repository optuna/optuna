from __future__ import annotations

from collections.abc import Callable
import datetime
from io import BytesIO
from typing import Any

import _pytest.capture
import pytest

import optuna
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.study.study import Study
from optuna.trial import TrialState
from optuna.visualization import plot_timeline as plotly_plot_timeline
from optuna.visualization._plotly_imports import _imports as plotly_imports
from optuna.visualization._timeline import _get_timeline_info
from optuna.visualization.matplotlib import plot_timeline as plt_plot_timeline
from optuna.visualization.matplotlib._matplotlib_imports import _imports as plt_imports


if plotly_imports.is_successful():
    from optuna.visualization._plotly_imports import go

if plt_imports.is_successful():
    from optuna.visualization.matplotlib._matplotlib_imports import plt


parametrize_plot_timeline = pytest.mark.parametrize(
    "plot_timeline",
    [plotly_plot_timeline, plt_plot_timeline],
)


def _create_study(
    trial_states: list[TrialState],
    trial_sys_attrs: dict[str, Any] | None = None,
) -> Study:
    study = optuna.create_study()
    fmax = float(len(trial_states))
    for i, s in enumerate(trial_states):
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": float(i)},
                distributions={"x": optuna.distributions.FloatDistribution(-1.0, fmax)},
                value=0.0 if s == TrialState.COMPLETE else None,
                state=s,
                system_attrs=trial_sys_attrs,
            )
        )
    return study


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


@pytest.mark.parametrize(
    "n_recent_trials, expected_count",
    [
        (None, 4),
        (2, 2),
        (100, 4),
    ],
)
def test_get_timeline_info_n_recent_trials(
    n_recent_trials: int | None, expected_count: int
) -> None:
    states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL, TrialState.RUNNING]
    study = _create_study(states)
    info = _get_timeline_info(study, n_recent_trials=n_recent_trials)

    assert len(info.bars) == expected_count

    if n_recent_trials is not None and n_recent_trials > 0 and expected_count > 0:
        all_trials = study.get_trials(deepcopy=False)
        expected_trials = all_trials[-expected_count:]
        for bar, trial in zip(info.bars, expected_trials):
            assert bar.number == trial.number


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


@parametrize_plot_timeline
@pytest.mark.parametrize(
    "trial_states, n_recent_trials",
    [
        ([], None),
        ([TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL, TrialState.RUNNING], None),
        ([TrialState.RUNNING, TrialState.FAIL, TrialState.PRUNED, TrialState.COMPLETE], None),
        ([TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL, TrialState.RUNNING], 2),
        ([TrialState.RUNNING, TrialState.FAIL, TrialState.PRUNED, TrialState.COMPLETE], 1),
        ([TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL], 5),  # More than available.
    ],
)
def test_get_timeline_plot(
    plot_timeline: Callable[..., Any],
    trial_states: list[TrialState],
    n_recent_trials: int | None,
) -> None:
    study = _create_study(trial_states)
    figure = plot_timeline(study, n_recent_trials=n_recent_trials)

    if isinstance(figure, go.Figure):
        figure.write_image(BytesIO())
    else:
        plt.savefig(BytesIO())
        plt.close()


@parametrize_plot_timeline
@pytest.mark.parametrize(
    "n_recent_trials",
    [0, -1, -10],
)
def test_plot_timeline_n_recent_trials_invalid(
    plot_timeline: Callable[..., Any],
    n_recent_trials: int | None,
) -> None:
    states = [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL, TrialState.RUNNING]
    study = _create_study(states)
    with pytest.raises(ValueError):
        plot_timeline(study, n_recent_trials=n_recent_trials)
