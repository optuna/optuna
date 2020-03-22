import py
from unittest.mock import patch

import optuna
from optuna.integration.mlflow import MlflowCallback
from ..test_structs import _create_frozen_trial


def test_happy_case(tmpdir):
    # type: (py.path.local) -> None

    db_file_name = "file:{}".format(tmpdir)

    mlflc = MlflowCallback(
        tracking_uri=db_file_name, metric_name="my_metric", experiment="my_experiment"
    )
    study = optuna.create_study()
    ft = _create_frozen_trial()
    mlflc(study, ft)


@patch("mlflow.set_tags")
def test_set_tags(set_tags, tmpdir):
    # type: (unittest.mock.MagicMock, py.path.local) -> None

    db_file_name = "file:{}".format(tmpdir)

    mlflc = MlflowCallback(
        tracking_uri=db_file_name, metric_name="my_metric", experiment="my_experiment"
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
    # type: (unittest.mock.MagicMock, py.path.local) -> None

    db_file_name = "file:{}".format(tmpdir)

    mlflc = MlflowCallback(
        tracking_uri=db_file_name, metric_name="my_metric", experiment="my_experiment"
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
    # type: (unittest.mock.MagicMock, py.path.local) -> None

    db_file_name = "file:{}".format(tmpdir)

    mlflc = MlflowCallback(
        tracking_uri=db_file_name, metric_name="my_metric", experiment="my_experiment"
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
    # type: (unittest.mock.MagicMock, py.path.local) -> None

    db_file_name = "file:{}".format(tmpdir)

    mlflc = MlflowCallback(tracking_uri=db_file_name, experiment="my_experiment")
    study = optuna.create_study()
    ft = _create_frozen_trial()
    mlflc(study, ft)

    assert log_metric.called
    assert log_metric.call_count == 1
    call_args = log_metric.call_args_list[0][0]
    assert call_args[0] == "trial_value"
    assert call_args[1] == 0.2
