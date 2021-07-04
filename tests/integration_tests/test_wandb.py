from unittest import mock

import optuna
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.integration import WeightsAndBiasesCallback


def _objective_func(trial: optuna.trial.Trial) -> float:

    x = trial.suggest_uniform("x", low=-10, high=10)
    y = trial.suggest_loguniform("y", low=1, high=10)
    return (x - 2) ** 2 + (y - 25) ** 2


@mock.patch("optuna.integration.wandb.wandb")
def test_run_initialized(wandb: mock.MagicMock) -> None:

    wandb.run = None
    WeightsAndBiasesCallback(
        metric_name="mse",
        project_name="optuna",
        group_name="summary",
        job_type="logging",
        mode="offline",
    )

    wandb.init.assert_called_once_with(
        project="optuna",
        group="summary",
        entity=None,
        name=None,
        job_type="logging",
        mode="offline",
        reinit=True,
    )


@mock.patch("optuna.integration.wandb.wandb")
def test_attributes_set_on_epoch(wandb: mock.MagicMock) -> None:

    wandb.run = None
    wandb.config.update = mock.MagicMock()

    wandbc = WeightsAndBiasesCallback()
    study = optuna.create_study(direction="minimize")
    study.optimize(_objective_func, n_trials=1, callbacks=[wandbc])

    expected = {
        "direction": "MINIMIZE",
        "trial_state": "COMPLETE",
        "distributions": {
            "x": UniformDistribution(high=10.0, low=-10.0),
            "y": LogUniformDistribution(high=10.0, low=1.0),
        },
    }

    wandb.config.update.assert_called_once_with(expected)


@mock.patch("optuna.integration.wandb.wandb")
def test_user_attributes_set_on_epoch(wandb: mock.MagicMock) -> None:

    wandb.run = None
    wandb.config.update = mock.MagicMock()

    wandbc = WeightsAndBiasesCallback()
    study = optuna.create_study(direction="minimize")
    study.set_user_attr("contributors", ["Akiba", "Sano"])
    study.set_user_attr("dataset", "MNIST")
    study.optimize(_objective_func, n_trials=1, callbacks=[wandbc])

    expected = {
        "direction": "MINIMIZE",
        "trial_state": "COMPLETE",
        "distributions": {
            "x": UniformDistribution(high=10.0, low=-10.0),
            "y": LogUniformDistribution(high=10.0, low=1.0),
        },
        "contributors": ["Akiba", "Sano"],
        "dataset": "MNIST",
    }

    wandb.config.update.assert_called_once_with(expected)


@mock.patch("optuna.integration.wandb.wandb")
def test_log_api_call_count(wandb: mock.Mock) -> None:

    wandb.run = None
    wandb.log = mock.MagicMock()

    wandbc = WeightsAndBiasesCallback()
    target_n_trials = 10
    study = optuna.create_study()
    study.optimize(_objective_func, n_trials=target_n_trials, callbacks=[wandbc])
    assert wandb.log.call_count == target_n_trials


@mock.patch("optuna.integration.wandb.wandb")
def test_values_registered_on_epoch(wandb: mock.Mock) -> None:

    wandb.run = None
    wandb.log = mock.MagicMock()

    wandbc = WeightsAndBiasesCallback()
    study = optuna.create_study()
    study.optimize(_objective_func, n_trials=1, callbacks=[wandbc])

    kall = wandb.log.call_args
    assert list(kall[0][0].keys()) == ["x", "y", "value"]
    assert kall[1] == {"step": 0}
