import datetime
import py
from unittest.mock import MagicMock
from unittest.mock import patch

import optuna
from optuna.distributions import UniformDistribution
from optuna.integration.mlflow import MLflowCallback
from optuna.structs import FrozenTrial
from optuna.structs import TrialState


def _create_frozen_trial():
    # type: () -> FrozenTrial

    return FrozenTrial(
        number=0,
        trial_id=0,
        state=TrialState.COMPLETE,
        value=0.2,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now(),
        params={"x": 10},
        distributions={"x": UniformDistribution(5, 12)},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
    )


def _objective_func(trial, x_max=1.0):
    # type: (optuna.trial.Trial, float) -> float

    x = trial.suggest_uniform("x", -x_max, x_max)
    y = trial.suggest_loguniform("y", 20, 30)
    z = trial.suggest_categorical("z", (-1.0, 1.0))
    assert isinstance(z, float)
    return (x - 2) ** 2 + (y - 25) ** 2 + z


def test_happy_case(tmpdir):
    # type: (py.path.local) -> None

    tracking_file_name = "file:{}".format(tmpdir)

    mlflc = MLflowCallback(
        tracking_uri=tracking_file_name, metric_name="my_metric", experiment="my_experiment"
    )
    study = optuna.create_study()
    ft = _create_frozen_trial()
    mlflc(study, ft)


@patch("mlflow.set_tags")
def test_set_tags(set_tags, tmpdir):
    # type: (MagicMock, py.path.local) -> None

    tracking_file_name = "file:{}".format(tmpdir)

    mlflc = MLflowCallback(
        tracking_uri=tracking_file_name, metric_name="my_metric", experiment="my_experiment"
    )
    study = optuna.create_study()
    ft = _create_frozen_trial()
    mlflc(study, ft)

    assert set_tags.called
    assert set_tags.call_count == 1
    call_arg = set_tags.call_args_list[0][0][0]
    assert call_arg["trial_state"] == "TrialState.COMPLETE"
    assert call_arg["x"] == "UniformDistribution(high=12, low=5)"
    assert call_arg["trial_number"] == "0"


@patch("mlflow.log_params")
def test_log_params(log_params, tmpdir):
    # type: (MagicMock, py.path.local) -> None

    tracking_file_name = "file:{}".format(tmpdir)

    mlflc = MLflowCallback(
        tracking_uri=tracking_file_name, metric_name="my_metric", experiment="my_experiment"
    )
    study = optuna.create_study()
    ft = _create_frozen_trial()
    mlflc(study, ft)

    assert log_params.called
    assert log_params.call_count == 1
    call_arg = log_params.call_args_list[0][0][0]
    assert call_arg["x"] == 10


@patch("mlflow.log_metric")
def test_log_metric_with_metric_name(log_metric, tmpdir):
    # type: (MagicMock, py.path.local) -> None

    tracking_file_name = "file:{}".format(tmpdir)

    mlflc = MLflowCallback(
        tracking_uri=tracking_file_name, metric_name="my_metric", experiment="my_experiment"
    )
    study = optuna.create_study()
    ft = _create_frozen_trial()
    mlflc(study, ft)

    assert log_metric.called
    assert log_metric.call_count == 1
    call_args = log_metric.call_args_list[0][0]
    assert call_args[0] == "my_metric"
    assert call_args[1] == 0.2


@patch("mlflow.log_metric")
def test_log_metric_with_default_metric_name(log_metric, tmpdir):
    # type: (MagicMock, py.path.local) -> None

    tracking_file_name = "file:{}".format(tmpdir)

    mlflc = MLflowCallback(tracking_uri=tracking_file_name, experiment="my_experiment")
    study = optuna.create_study()
    ft = _create_frozen_trial()
    mlflc(study, ft)

    assert log_metric.called
    assert log_metric.call_count == 1
    call_args = log_metric.call_args_list[0][0]
    assert call_args[0] == "trial_value"
    assert call_args[1] == 0.2


@patch("mlflow.log_metric")
@patch("mlflow.log_params")
@patch("mlflow.set_tags")
def test_end_to_end(set_tags, log_params, log_metric, tmpdir):
    # type: (MagicMock, MagicMock, MagicMock, py.path.local) -> None

    tracking_file_name = "file:{}".format(tmpdir)

    mlflc = MLflowCallback(tracking_uri=tracking_file_name)
    study = optuna.create_study(study_name="my_study")
    study.optimize(_objective_func, n_trials=10, callbacks=[mlflc])

    assert set_tags.called
    assert set_tags.call_count == 10
    call_arg = set_tags.call_args_list[0][0][0]
    assert call_arg["trial_state"] == "TrialState.COMPLETE"
    assert call_arg["trial_number"] == "0"

    assert log_params.called
    assert log_params.call_count == 10
    call_arg = log_params.call_args_list[0][0][0]
    assert "x" in call_arg
    assert "y" in call_arg
    assert "z" in call_arg

    assert log_metric.called
    assert log_metric.call_count == 10
    call_args = log_metric.call_args_list[0][0]
    assert call_args[0] == "trial_value"
