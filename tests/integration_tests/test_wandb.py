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


@mock.patch("optuna.integration.wandb.wandb")
def test_run_initialized(wandb: mock.MagicMock) -> None:

    wandb_kwargs = {
        "project": "optuna",
        "group": "summary",
        "job_type": "logging",
        "mode": "offline",
    }

    WeightsAndBiasesCallback(
        metric_name="mse",
        wandb_kwargs=wandb_kwargs,
    )

    wandb.init.assert_called_once_with(
        project="optuna",
        group="summary",
        job_type="logging",
        mode="offline",
    )


@mock.patch("optuna.integration.wandb.wandb")
def test_attributes_set_on_epoch(wandb: mock.MagicMock) -> None:

    wandb.config.update = mock.MagicMock()

    wandbc = WeightsAndBiasesCallback()
    study = optuna.create_study(direction="minimize")
    study.optimize(_objective_func, n_trials=1, callbacks=[wandbc])

    expected = {"direction": ["MINIMIZE"]}
    wandb.config.update.assert_called_once_with(expected)


@mock.patch("optuna.integration.wandb.wandb")
def test_multiobjective_attributes_set_on_epoch(wandb: mock.MagicMock) -> None:

    wandb.config.update = mock.MagicMock()

    wandbc = WeightsAndBiasesCallback()
    study = optuna.create_study(directions=["minimize", "maximize"])
    study.optimize(_multiobjective_func, n_trials=1, callbacks=[wandbc])

    expected = {"direction": ["MINIMIZE", "MAXIMIZE"]}
    wandb.config.update.assert_called_once_with(expected)


@mock.patch("optuna.integration.wandb.wandb")
def test_log_api_call_count(wandb: mock.Mock) -> None:

    wandb.log = mock.MagicMock()

    wandbc = WeightsAndBiasesCallback()
    target_n_trials = 10
    study = optuna.create_study()
    study.optimize(_objective_func, n_trials=target_n_trials, callbacks=[wandbc])
    assert wandb.log.call_count == target_n_trials


@pytest.mark.parametrize(
    "metric,expected", [("value", ["x", "y", "value"]), ("foo", ["x", "y", "foo"])]
)
@mock.patch("optuna.integration.wandb.wandb")
def test_values_registered_on_epoch(wandb: mock.Mock, metric: str, expected: List[str]) -> None:

    wandb.log = mock.MagicMock()

    wandbc = WeightsAndBiasesCallback(metric_name=metric)
    study = optuna.create_study()
    study.optimize(_objective_func, n_trials=1, callbacks=[wandbc])

    kall = wandb.log.call_args
    assert list(kall[0][0].keys()) == expected
    assert kall[1] == {"step": 0}


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
    wandb: mock.Mock, metrics: Union[str, Sequence[str]], expected: List[str]
) -> None:

    wandb.log = mock.MagicMock()

    wandbc = WeightsAndBiasesCallback(metric_name=metrics)
    study = optuna.create_study(directions=["minimize", "maximize"])
    study.optimize(_multiobjective_func, n_trials=1, callbacks=[wandbc])

    kall = wandb.log.call_args
    assert list(kall[0][0].keys()) == expected
    assert kall[1] == {"step": 0}


@pytest.mark.parametrize("metrics", [["foo"], ["foo", "bar", "baz"]])
@mock.patch("optuna.integration.wandb.wandb")
def test_multiobjective_raises_on_name_mismatch(wandb: mock.MagicMock, metrics: List[str]) -> None:

    wandbc = WeightsAndBiasesCallback(metric_name=metrics)
    study = optuna.create_study(directions=["minimize", "maximize"])

    with pytest.raises(ValueError):
        study.optimize(_multiobjective_func, n_trials=1, callbacks=[wandbc])


@pytest.mark.parametrize("metrics", [{0: "foo", 1: "bar"}])
def test_multiobjective_raises_on_type_mismatch(metrics: Any) -> None:

    with pytest.raises(TypeError):
        WeightsAndBiasesCallback(metric_name=metrics)
