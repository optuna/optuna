import copy
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

    wandb.sdk.wandb_run.Run = mock.MagicMock

    wandb_kwargs = {
        "project": "optuna",
        "group": "summary",
        "job_type": "logging",
        "mode": "offline",
        "tags": ["test-tag"],
    }

    WeightsAndBiasesCallback(metric_name="mse", wandb_kwargs=wandb_kwargs, as_multirun=False)
    wandb.init.assert_called_once_with(
        project="optuna", group="summary", job_type="logging", mode="offline", tags=["test-tag"]
    )

    wandbc = WeightsAndBiasesCallback(
        metric_name="mse", wandb_kwargs=wandb_kwargs, as_multirun=True
    )
    wandb.run = None

    study = optuna.create_study(direction="minimize")
    _wrapped_func = wandbc.track_in_wandb(_objective_func)
    study.optimize(_wrapped_func, n_trials=1, callbacks=[wandbc])

    wandb.init = mock.MagicMock()
    study_id = study._study_id
    frozen_trial = study.trials[0]
    trial_id = study._storage.get_trial_id_from_study_id_trial_number(
        study_id, frozen_trial.number
    )
    trial = optuna.Trial(study, trial_id)

    _ = _wrapped_func(trial)

    wandb.init.assert_called_once_with(
        project="optuna", group="summary", job_type="logging", mode="offline", tags=["test-tag"]
    )

    wandb.init = mock.MagicMock()
    study.optimize(_objective_func, n_trials=1, callbacks=[wandbc])
    wandb.init.assert_called_once_with(
        project="optuna", group="summary", job_type="logging", mode="offline", tags=["test-tag"]
    )


@mock.patch("optuna.integration.wandb.wandb")
def test_attributes_set_on_epoch(wandb: mock.MagicMock) -> None:

    # Vanilla update
    wandb.sdk.wandb_run.Run = mock.MagicMock

    study = optuna.create_study(direction="minimize", study_name="test_study")
    wandbc = WeightsAndBiasesCallback()
    study.optimize(_objective_func, n_trials=1, callbacks=[wandbc])

    expected = {"direction": ["MINIMIZE"]}
    wandb.run.config.update.call_args[0][0].pop("x")
    wandb.run.config.update.call_args[0][0].pop("y")
    wandb.run.config.update.assert_called_once_with(expected)

    wandbc = WeightsAndBiasesCallback(as_multirun=True)
    wandb.run = mock.MagicMock()
    _wrapped_func = wandbc.track_in_wandb(_objective_func)
    study.optimize(_wrapped_func, n_trials=1, callbacks=[wandbc])
    wandb.run.config.update.call_args[0][0].pop("x")
    wandb.run.config.update.call_args[0][0].pop("y")
    wandb.run.config.update.assert_called_once_with(expected)

    wandb.run = None
    study.optimize(_objective_func, n_trials=1, callbacks=[wandbc])
    wandb.init().config.update.call_args[0][0].pop("x")
    wandb.init().config.update.call_args[0][0].pop("y")
    wandb.init().config.update.assert_called_once_with(expected)


@mock.patch("optuna.integration.wandb.wandb")
def test_multiobjective_attributes_set_on_epoch(wandb: mock.MagicMock) -> None:

    wandb.sdk.wandb_run.Run = mock.MagicMock

    study = optuna.create_study(directions=["minimize", "maximize"])
    wandbc = WeightsAndBiasesCallback()
    study.optimize(_multiobjective_func, n_trials=1, callbacks=[wandbc])

    expected = {"direction": ["MINIMIZE", "MAXIMIZE"]}
    wandb.run.config.update.call_args[0][0].pop("x")
    wandb.run.config.update.call_args[0][0].pop("y")
    wandb.run.config.update.assert_called_once_with(expected)

    wandbc = WeightsAndBiasesCallback(as_multirun=True)
    wandb.run = mock.MagicMock()
    _wrapped_func = wandbc.track_in_wandb(_multiobjective_func)
    study.optimize(_wrapped_func, n_trials=1, callbacks=[wandbc])
    wandb.run.config.update.call_args[0][0].pop("x")
    wandb.run.config.update.call_args[0][0].pop("y")
    wandb.run.config.update.assert_called_once_with(expected)

    wandb.run = None
    study.optimize(_multiobjective_func, n_trials=1, callbacks=[wandbc])
    wandb.init().config.update.call_args[0][0].pop("x")
    wandb.init().config.update.call_args[0][0].pop("y")
    wandb.init().config.update.assert_called_once_with(expected)


@mock.patch("optuna.integration.wandb.wandb")
def test_log_api_call_count(wandb: mock.MagicMock) -> None:

    wandb.sdk.wandb_run.Run = mock.MagicMock

    study = optuna.create_study()
    wandbc = WeightsAndBiasesCallback()
    target_n_trials = 10
    study.optimize(_objective_func, n_trials=target_n_trials, callbacks=[wandbc])
    assert wandb.run.log.call_count == target_n_trials

    wandbc = WeightsAndBiasesCallback(as_multirun=True)
    wandb.run = mock.MagicMock()
    _wrapped_func = wandbc.track_in_wandb(_objective_func)
    study.optimize(_wrapped_func, n_trials=target_n_trials, callbacks=[wandbc])

    assert wandb.run.log.call_count == target_n_trials

    wandb.run = None
    study.optimize(_objective_func, n_trials=target_n_trials, callbacks=[wandbc])
    assert wandb.init().log.call_count == target_n_trials


@pytest.mark.parametrize(
    "metric,expected", [("value", ["x", "y", "value"]), ("foo", ["x", "y", "foo"])]
)
@mock.patch("optuna.integration.wandb.wandb")
def test_values_registered_on_epoch(
    wandb: mock.MagicMock, metric: str, expected: List[str]
) -> None:
    def assert_call_args(log_func: mock.MagicMock, regular: bool) -> None:
        kall = log_func.call_args
        assert list(kall[0][0].keys()) == expected
        assert kall[1] == {"step": 0 if regular else None}

    wandb.sdk.wandb_run.Run = mock.MagicMock

    base_study = optuna.create_study()

    study = copy.deepcopy(base_study)
    wandbc = WeightsAndBiasesCallback(metric_name=metric)
    study.optimize(_objective_func, n_trials=1, callbacks=[wandbc])
    assert_call_args(wandb.run.log, bool(wandb.run))

    study = copy.deepcopy(base_study)
    wandb.run = mock.MagicMock()
    wandbc = WeightsAndBiasesCallback(metric_name=metric, as_multirun=True)
    _wrapped_func = wandbc.track_in_wandb(_objective_func)
    study.optimize(_wrapped_func, n_trials=1, callbacks=[wandbc])
    assert_call_args(wandb.run.log, bool(wandb.run))

    study = copy.deepcopy(base_study)
    wandb.run = None
    study.optimize(_objective_func, n_trials=1, callbacks=[wandbc])
    assert_call_args(wandb.init().log, bool(wandb.run))


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
    def assert_call_args(log_func: mock.MagicMock, regular: bool) -> None:

        kall = log_func.call_args
        assert list(kall[0][0].keys()) == expected

        assert kall[1] == {"step": 0 if regular else None}

    wandb.sdk.wandb_run.Run = mock.MagicMock

    base_study = optuna.create_study(directions=["minimize", "maximize"])

    study = copy.deepcopy(base_study)
    wandbc = WeightsAndBiasesCallback(metric_name=metrics)
    study.optimize(_multiobjective_func, n_trials=1, callbacks=[wandbc])
    assert_call_args(wandb.run.log, bool(wandb.run))

    study = copy.deepcopy(base_study)
    wandbc = WeightsAndBiasesCallback(as_multirun=True, metric_name=metrics)
    wandb.run = mock.MagicMock()
    _wrapped_func = wandbc.track_in_wandb(_multiobjective_func)
    study.optimize(_wrapped_func, n_trials=1, callbacks=[wandbc])
    assert_call_args(wandb.run.log, bool(wandb.run))

    wandb.run = None
    study = copy.deepcopy(base_study)
    study.optimize(_multiobjective_func, n_trials=1, callbacks=[wandbc])
    assert_call_args(wandb.init().log, bool(wandb.run))


@pytest.mark.parametrize("metrics", [["foo"], ["foo", "bar", "baz"]])
@mock.patch("optuna.integration.wandb.wandb")
def test_multiobjective_raises_on_name_mismatch(wandb: mock.MagicMock, metrics: List[str]) -> None:

    wandb.sdk.wandb_run.Run = mock.MagicMock

    study = optuna.create_study(directions=["minimize", "maximize"])
    wandbc = WeightsAndBiasesCallback(metric_name=metrics)

    with pytest.raises(ValueError):
        study.optimize(_multiobjective_func, n_trials=1, callbacks=[wandbc])


@pytest.mark.parametrize("metrics", [{0: "foo", 1: "bar"}])
def test_multiobjective_raises_on_type_mismatch(metrics: Any) -> None:

    with pytest.raises(TypeError):
        WeightsAndBiasesCallback(metric_name=metrics)
