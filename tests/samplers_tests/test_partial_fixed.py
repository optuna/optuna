from collections.abc import Callable
from unittest.mock import patch
import warnings

import pytest

import optuna
from optuna.samplers import BaseSampler
from optuna.samplers import PartialFixedSampler
from optuna.samplers import RandomSampler
from optuna.trial import Trial


parametrize_sampler = pytest.mark.parametrize(
    "sampler_class",
    [
        optuna.samplers.RandomSampler,
        lambda: optuna.samplers.TPESampler(n_startup_trials=0),
        lambda: optuna.samplers.TPESampler(n_startup_trials=0, multivariate=True),
        lambda: optuna.samplers.TPESampler(n_startup_trials=0, multivariate=True, group=True),
        lambda: optuna.samplers.CmaEsSampler(n_startup_trials=0),
        lambda: optuna.samplers.CmaEsSampler(n_startup_trials=0, use_separable_cma=True),
        optuna.samplers.NSGAIISampler,
        optuna.samplers.NSGAIIISampler,
        optuna.samplers.QMCSampler,
        lambda: optuna.samplers.GPSampler(n_startup_trials=0),
        lambda: optuna.samplers.GPSampler(n_startup_trials=0, deterministic_objective=True),
    ],
)


@parametrize_sampler
def test_fixed_sampling(sampler_class: Callable[[], BaseSampler]) -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        z = trial.suggest_float("z", -10, 10)
        return x**2 + y**2 + z**2

    base_sampler = sampler_class()

    study = optuna.create_study(sampler=base_sampler)
    study.optimize(objective, n_trials=1)

    # Fix parameter ``z`` as 0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = PartialFixedSampler(fixed_params={"z": 0}, base_sampler=base_sampler)
    study.sampler = sampler
    study.optimize(objective, n_trials=1)

    assert study.trials[1].params["z"] == 0


def test_float_to_int() -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_int("x", -10, 10)
        y = trial.suggest_int("y", -10, 10)
        return x**2 + y**2

    fixed_y = 0.5

    # Parameters of Int-type-distribution are rounded to int-type,
    # even if they are defined as float-type.
    study = optuna.create_study()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        study.sampler = PartialFixedSampler(
            fixed_params={"y": fixed_y}, base_sampler=study.sampler
        )
    # Since `fixed_y` is out-of-the-range value in the corresponding suggest_int,
    # `UserWarning` will occur.
    with pytest.warns(UserWarning):
        study.optimize(objective, n_trials=1)
    assert study.trials[0].params["y"] == int(fixed_y)


@pytest.mark.parametrize("fixed_y", [-2, 2])
def test_out_of_the_range_numerical(fixed_y: int) -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_int("x", -1, 1)
        y = trial.suggest_int("y", -1, 1)
        return x**2 + y**2

    # It is possible to fix numerical parameters as out-of-the-range value.
    # `UserWarning` will occur.
    study = optuna.create_study()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        study.sampler = PartialFixedSampler(
            fixed_params={"y": fixed_y}, base_sampler=study.sampler
        )
    with pytest.warns(UserWarning):
        study.optimize(objective, n_trials=1)
    assert study.trials[0].params["y"] == fixed_y


def test_out_of_the_range_categorical() -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_int("x", -1, 1)
        y = trial.suggest_categorical("y", [-1, 0, 1])
        return x**2 + y**2

    fixed_y = 2

    # It isn't possible to fix categorical parameters as out-of-the-range value.
    # `ValueError` will occur.
    study = optuna.create_study()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        study.sampler = PartialFixedSampler(
            fixed_params={"y": fixed_y}, base_sampler=study.sampler
        )
    with pytest.raises(ValueError):
        study.optimize(objective, n_trials=1)


def test_partial_fixed_experimental_warning() -> None:
    study = optuna.create_study()
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        optuna.samplers.PartialFixedSampler(fixed_params={"x": 0}, base_sampler=study.sampler)


def test_call_after_trial_of_base_sampler() -> None:
    base_sampler = RandomSampler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = PartialFixedSampler(fixed_params={}, base_sampler=base_sampler)
    study = optuna.create_study(sampler=sampler)
    with patch.object(base_sampler, "after_trial", wraps=base_sampler.after_trial) as mock_object:
        study.optimize(lambda _: 1.0, n_trials=1)
        assert mock_object.call_count == 1


def test_fixed_none_value_sampling() -> None:
    def objective(trial: Trial) -> float:
        trial.suggest_categorical("x", (None, 0))
        return 0.0

    tpe = optuna.samplers.TPESampler()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        # In this following case , "x" should sample only `None`
        sampler = optuna.samplers.PartialFixedSampler(fixed_params={"x": None}, base_sampler=tpe)

    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10)

    for trial in study.trials:
        assert trial.params["x"] is None
