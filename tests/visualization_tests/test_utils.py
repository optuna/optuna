import datetime
import logging
from textwrap import dedent
from typing import cast

import numpy as np
import pytest
from pytest import LogCaptureFixture

import optuna
from optuna.distributions import FloatDistribution
from optuna.study import create_study
from optuna.testing.visualization import prepare_study_with_trials
from optuna.trial import create_trial
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.visualization import is_available
from optuna.visualization._utils import _check_plot_args
from optuna.visualization._utils import _filter_nonfinite
from optuna.visualization._utils import _is_log_scale
from optuna.visualization._utils import _make_hovertext


def test_is_log_scale() -> None:
    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"param_linear": 1.0},
            distributions={"param_linear": FloatDistribution(0.0, 3.0)},
        )
    )
    study.add_trial(
        create_trial(
            value=2.0,
            params={"param_linear": 2.0, "param_log": 1e-3},
            distributions={
                "param_linear": FloatDistribution(0.0, 3.0),
                "param_log": FloatDistribution(1e-5, 1.0, log=True),
            },
        )
    )
    assert _is_log_scale(study.trials, "param_log")
    assert not _is_log_scale(study.trials, "param_linear")


def _is_plotly_available() -> bool:
    try:
        import plotly  # NOQA

        available = True
    except Exception:
        available = False
    return available


def test_visualization_is_available() -> None:
    assert is_available() == _is_plotly_available()


def test_check_plot_args() -> None:
    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        _check_plot_args(study, None, "Objective Value")

    with pytest.warns(UserWarning):
        _check_plot_args(study, lambda t: cast(float, t.value), "Objective Value")


@pytest.mark.parametrize("value, expected", [(float("inf"), 1), (-float("inf"), 1), (0.0, 2)])
def test_filter_inf_trials(value: float, expected: int) -> None:
    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"x": 1.0},
            distributions={"x": FloatDistribution(0.0, 1.0)},
        )
    )
    study.add_trial(
        create_trial(
            value=value,
            params={"x": 0.0},
            distributions={"x": FloatDistribution(0.0, 1.0)},
        )
    )

    trials = _filter_nonfinite(study.get_trials(states=(TrialState.COMPLETE,)))
    assert len(trials) == expected
    assert all([t.number == num for t, num in zip(trials, range(expected))])


@pytest.mark.parametrize(
    "value,objective_selected,expected",
    [
        (float("inf"), 0, 2),
        (-float("inf"), 0, 2),
        (0.0, 0, 3),
        (float("inf"), 1, 1),
        (-float("inf"), 1, 1),
        (0.0, 1, 3),
    ],
)
def test_filter_inf_trials_multiobjective(
    value: float, objective_selected: int, expected: int
) -> None:
    study = create_study(directions=["minimize", "maximize"])
    study.add_trial(
        create_trial(
            values=[0.0, 1.0],
            params={"x": 1.0},
            distributions={"x": FloatDistribution(0.0, 1.0)},
        )
    )
    study.add_trial(
        create_trial(
            values=[0.0, value],
            params={"x": 0.0},
            distributions={"x": FloatDistribution(0.0, 1.0)},
        )
    )
    study.add_trial(
        create_trial(
            values=[value, value],
            params={"x": 0.0},
            distributions={"x": FloatDistribution(0.0, 1.0)},
        )
    )

    def _target(t: FrozenTrial) -> float:
        return t.values[objective_selected]

    trials = _filter_nonfinite(study.get_trials(states=(TrialState.COMPLETE,)), target=_target)
    assert len(trials) == expected
    assert all([t.number == num for t, num in zip(trials, range(expected))])


@pytest.mark.parametrize("with_message", [True, False])
def test_filter_inf_trials_message(caplog: LogCaptureFixture, with_message: bool) -> None:
    study = create_study()
    study.add_trial(
        create_trial(
            value=0.0,
            params={"x": 1.0},
            distributions={"x": FloatDistribution(0.0, 1.0)},
        )
    )
    study.add_trial(
        create_trial(
            value=float("inf"),
            params={"x": 0.0},
            distributions={"x": FloatDistribution(0.0, 1.0)},
        )
    )

    optuna.logging.enable_propagation()
    _filter_nonfinite(study.get_trials(states=(TrialState.COMPLETE,)), with_message=with_message)
    msg = "Trial 1 is omitted in visualization because its objective value is inf or nan."

    if with_message:
        assert msg in caplog.text
        n_filtered_as_inf = 0
        for record in caplog.records:
            if record.msg == msg:
                assert record.levelno == logging.WARNING
                n_filtered_as_inf += 1
        assert n_filtered_as_inf == 1
    else:
        assert msg not in caplog.text


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_filter_nonfinite_with_invalid_target() -> None:
    study = prepare_study_with_trials()
    trials = study.get_trials(states=(TrialState.COMPLETE,))
    with pytest.raises(ValueError):
        _filter_nonfinite(trials, target=lambda t: "invalid target")  # type: ignore


def test_make_hovertext() -> None:
    trial_no_user_attrs = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 10},
        distributions={"x": FloatDistribution(5, 12)},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )
    assert (
        _make_hovertext(trial_no_user_attrs)
        == dedent(
            """
        {
          "number": 0,
          "values": [
            0.2
          ],
          "params": {
            "x": 10
          }
        }
        """
        )
        .strip()
        .replace("\n", "<br>")
    )

    trial_user_attrs_valid_json = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 10},
        distributions={"x": FloatDistribution(5, 12)},
        user_attrs={"a": 42, "b": 3.14},
        system_attrs={},
        intermediate_values={},
    )
    assert (
        _make_hovertext(trial_user_attrs_valid_json)
        == dedent(
            """
        {
          "number": 0,
          "values": [
            0.2
          ],
          "params": {
            "x": 10
          },
          "user_attrs": {
            "a": 42,
            "b": 3.14
          }
        }
        """
        )
        .strip()
        .replace("\n", "<br>")
    )

    trial_user_attrs_invalid_json = FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 10},
        distributions={"x": FloatDistribution(5, 12)},
        user_attrs={"a": 42, "b": 3.14, "c": np.zeros(1), "d": np.nan},
        system_attrs={},
        intermediate_values={},
    )
    assert (
        _make_hovertext(trial_user_attrs_invalid_json)
        == dedent(
            """
        {
          "number": 0,
          "values": [
            0.2
          ],
          "params": {
            "x": 10
          },
          "user_attrs": {
            "a": 42,
            "b": 3.14,
            "c": "[0.]",
            "d": NaN
          }
        }
        """
        )
        .strip()
        .replace("\n", "<br>")
    )
