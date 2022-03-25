from typing import Any
from typing import List
from typing import Sequence
from typing import Tuple
from typing import Union
from unittest import mock

import pytest

import optuna
from optuna.integration import WeightsAndBiasesCallback


def _objective_func(trial: optuna.trial.Trial) -> float:

    x = trial.suggest_float("x", low=-10, high=10)
    y = trial.suggest_float("y", low=1, high=10, log=True)
    return (x - 2) ** 2 + (y - 25) ** 2


def _multiobjective_func(trial: optuna.trial.Trial) -> Tuple[float, float]:

    x = trial.suggest_float("x", low=-10, high=10)
    y = trial.suggest_float("y", low=1, high=10, log=True)
    first_objective = (x - 2) ** 2 + (y - 25) ** 2
    second_objective = (x - 2) ** 3 + (y - 25) ** 3

    return first_objective, second_objective


def _objective_func_with_run_id(trial: optuna.trial.Trial) -> float:

    trial.set_user_attr("run_id", "test-id")
    return _objective_func(trial)


def _multiobjective_func_with_run_id(trial: optuna.trial.Trial) -> Tuple[float, float]:

    trial.set_user_attr("run_id", "test-id")
    return _multiobjective_func(trial)


@mock.patch("optuna.integration.wandb.wandb")
def test_run_initialized(wandb: mock.MagicMock) -> None:

    wandb.Api().settings.__getitem__.return_value = "mock"
    wandb.sdk.wandb_run.Run = mock.MagicMock

    wandb_kwargs = {
        "project": "optuna",
        "group": "summary",
        "job_type": "logging",
        "mode": "offline",
    }

    WeightsAndBiasesCallback(
        study_name="test_study", metric_name="mse", wandb_kwargs=wandb_kwargs, as_sweeps=False
    )

    wandb.init.assert_called_once_with(
        project="optuna", group="summary", job_type="logging", mode="offline", tags=["test_study"]
    )


@mock.patch("optuna.integration.wandb.wandb")
def test_attributes_set_on_epoch(wandb: mock.MagicMock) -> None:

    wandb.Api().settings.__getitem__.return_value = "mock"
    wandb.sdk.wandb_run.Run = mock.MagicMock

    study = optuna.create_study(direction="minimize")
    wandbc = WeightsAndBiasesCallback(study_name=study.study_name)
    study.optimize(_objective_func, n_trials=1, callbacks=[wandbc])

    expected = {"direction": ["MINIMIZE"]}
    wandb.run.config.update.assert_called_once_with(expected)

    wandb.run = None

    wandbc = WeightsAndBiasesCallback(study_name=study.study_name, as_sweeps=True)
    study.optimize(_objective_func, n_trials=1, callbacks=[wandbc])

    wandb.init().config.update.assert_called_once_with(expected)

    study.optimize(_objective_func_with_run_id, n_trials=1, callbacks=[wandbc])

    wandb.Api().run().config.update.assert_called_once_with(expected)


@mock.patch("optuna.integration.wandb.wandb")
def test_multiobjective_attributes_set_on_epoch(wandb: mock.MagicMock) -> None:

    wandb.Api().settings.__getitem__.return_value = "mock"
    wandb.sdk.wandb_run.Run = mock.MagicMock

    study = optuna.create_study(directions=["minimize", "maximize"])
    wandbc = WeightsAndBiasesCallback(study_name=study.study_name)
    study.optimize(_multiobjective_func, n_trials=1, callbacks=[wandbc])

    expected = {"direction": ["MINIMIZE", "MAXIMIZE"]}
    wandb.run.config.update.assert_called_once_with(expected)

    wandb.run = None

    wandbc = WeightsAndBiasesCallback(study_name=study.study_name, as_sweeps=True)
    study.optimize(_multiobjective_func, n_trials=1, callbacks=[wandbc])
    wandb.init().config.update.assert_called_once_with(expected)

    study.optimize(_multiobjective_func_with_run_id, n_trials=1, callbacks=[wandbc])
    wandb.Api().run().config.update.assert_called_once_with(expected)


@mock.patch("optuna.integration.wandb.wandb")
def test_log_api_call_count(wandb: mock.MagicMock) -> None:

    wandb.Api().settings.__getitem__.return_value = "mock"
    wandb.sdk.wandb_run.Run = mock.MagicMock

    study = optuna.create_study()
    wandbc = WeightsAndBiasesCallback(study_name=study.study_name)
    target_n_trials = 10
    study.optimize(_objective_func, n_trials=target_n_trials, callbacks=[wandbc])
    assert wandb.run.log.call_count == target_n_trials

    wandb.run = None

    wandbc = WeightsAndBiasesCallback(study_name=study.study_name, as_sweeps=True)
    study.optimize(_objective_func, n_trials=target_n_trials, callbacks=[wandbc])
    assert wandb.init().log.call_count == target_n_trials

    study.optimize(_objective_func_with_run_id, n_trials=target_n_trials, callbacks=[wandbc])
    assert wandb.Api().run().summary.update.call_count == target_n_trials


@pytest.mark.parametrize(
    "metric,expected", [("value", ["x", "y", "value"]), ("foo", ["x", "y", "foo"])]
)
@mock.patch("optuna.integration.wandb.wandb")
def test_values_registered_on_epoch(
    wandb: mock.MagicMock, metric: str, expected: List[str]
) -> None:
    def assert_call_args(log_func: mock.MagicMock, regular: bool):
        kall = log_func.call_args
        assert list(kall[0][0].keys()) == expected

        if regular:
            assert kall[1] == {"step": 0}

    wandb.Api().settings.__getitem__.return_value = "mock"
    wandb.sdk.wandb_run.Run = mock.MagicMock

    study = optuna.create_study()
    wandbc = WeightsAndBiasesCallback(study_name=study.study_name, metric_name=metric)
    study.optimize(_objective_func, n_trials=1, callbacks=[wandbc])
    assert_call_args(wandb.run.log, bool(wandb.run))

    wandb.run = None

    wandbc = WeightsAndBiasesCallback(
        study_name=study.study_name, metric_name=metric, as_sweeps=True
    )
    study.optimize(_objective_func, n_trials=1, callbacks=[wandbc])
    assert_call_args(wandb.init().log, bool(wandb.run))

    study.optimize(_objective_func_with_run_id, n_trials=1, callbacks=[wandbc])
    assert_call_args(wandb.Api().run().summary.update, bool(wandb.run))


@pytest.mark.parametrize(
    "metrics,expected",
    [
        ("value", ["x", "y", "value_0", "value_1"]),
        (["foo", "bar"], ["x", "y", "foo", "bar"]),
        (("foo", "bar"), ["x", "y", "foo", "bar"]),
    ],
)
@mock.patch("optuna.integration.wandb.wandb")
def test_multiobjective_values_registered_on_epoch(
    wandb: mock.MagicMock, metrics: Union[str, Sequence[str]], expected: List[str]
) -> None:
    def assert_call_args(log_func: mock.MagicMock, regular: bool):

        kall = log_func.call_args
        assert list(kall[0][0].keys()) == expected

        if regular:
            assert kall[1] == {"step": 0}

    wandb.Api().settings.__getitem__.return_value = "mock"
    wandb.sdk.wandb_run.Run = mock.MagicMock

    study = optuna.create_study(directions=["minimize", "maximize"])
    wandbc = WeightsAndBiasesCallback(study_name=study.study_name, metric_name=metrics)
    study.optimize(_multiobjective_func, n_trials=1, callbacks=[wandbc])
    assert_call_args(wandb.run.log, bool(wandb.run))

    wandb.run = None

    wandbc = WeightsAndBiasesCallback(
        study_name=study.study_name, as_sweeps=True, metric_name=metrics
    )
    study.optimize(_multiobjective_func, n_trials=1, callbacks=[wandbc])
    assert_call_args(wandb.init().log, bool(wandb.run))

    study.optimize(_multiobjective_func_with_run_id, n_trials=1, callbacks=[wandbc])
    assert_call_args(wandb.Api().run().summary.update, bool(wandb.run))


@pytest.mark.parametrize("metrics", [["foo"], ["foo", "bar", "baz"]])
@mock.patch("optuna.integration.wandb.wandb")
def test_multiobjective_raises_on_name_mismatch(wandb: mock.MagicMock, metrics: List[str]) -> None:

    wandb.Api().settings.__getitem__.return_value = "mock"
    wandb.sdk.wandb_run.Run = mock.MagicMock

    study = optuna.create_study(directions=["minimize", "maximize"])
    wandbc = WeightsAndBiasesCallback(study_name=study.study_name, metric_name=metrics)

    with pytest.raises(ValueError):
        study.optimize(_multiobjective_func, n_trials=1, callbacks=[wandbc])


@pytest.mark.parametrize("metrics", [{0: "foo", 1: "bar"}])
def test_multiobjective_raises_on_type_mismatch(metrics: Any) -> None:

    with pytest.raises(TypeError):
        WeightsAndBiasesCallback(study_name="test_study", metric_name=metrics)
