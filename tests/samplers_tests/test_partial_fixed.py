from typing import Union
from unittest.mock import patch
import warnings

import pytest

import optuna
from optuna.samplers import PartialFixedSampler
from optuna.samplers import RandomSampler
from optuna.trial import Trial


def test_params_identity() -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_uniform("x", -10, 10)
        y = trial.suggest_uniform("y", -10, 10)
        return x ** 2 + y ** 2

    study0 = optuna.create_study()
    study0.sampler = RandomSampler(seed=42)
    study0.optimize(objective, n_trials=1)
    x_sampled0 = study0.trials[0].params["x"]

    # Fix parameter ``y`` as 0.
    study1 = optuna.create_study()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        study1.sampler = PartialFixedSampler(
            fixed_params={"y": 0}, base_sampler=RandomSampler(seed=42)
        )
    study1.optimize(objective, n_trials=1)
    x_sampled1 = study1.trials[0].params["x"]
    y_sampled1 = study1.trials[0].params["y"]
    assert x_sampled1 == x_sampled0
    assert y_sampled1 == 0


@pytest.mark.parametrize("fixed_y", [0.5, 5])
def test_incompatible_types(fixed_y: Union[float, int]) -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_int("x", -10, 10)
        y = trial.suggest_int("y", -10, 10)
        return x ** 2 + y ** 2

    # Parameters of Int-type-distribution are rounded to int-type,
    # even if they are specified as float-type.
    study = optuna.create_study()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        study.sampler = PartialFixedSampler(
            fixed_params={"y": fixed_y}, base_sampler=study.sampler
        )
    study.optimize(objective, n_trials=1)
    assert study.trials[0].params["y"] == int(fixed_y)


@pytest.mark.parametrize("fixed_y", [-2, 2])
def test_out_of_the_range(fixed_y: int) -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_int("x", -1, 1)
        y = trial.suggest_int("y", -1, 1)
        return x ** 2 + y ** 2

    # It is possible to fix parameters as out-of-the-range value.
    # Userwarnings will occur.
    study = optuna.create_study()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        study.sampler = PartialFixedSampler(
            fixed_params={"y": fixed_y}, base_sampler=study.sampler
        )
    with pytest.warns(UserWarning):
        study.optimize(objective, n_trials=1)
    assert study.trials[0].params["y"] == fixed_y


def test_partial_fixed_experimental_warning() -> None:
    study = optuna.create_study()
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        optuna.samplers.PartialFixedSampler(fixed_params=None, base_sampler=study.sampler)


def test_reseed_rng() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = PartialFixedSampler(fixed_params=None, base_sampler=RandomSampler)
    original_seed = sampler._rng.seed

    with patch.object(sampler, "reseed_rng", wraps=sampler.reseed_rng) as mock_object:
        sampler.reseed_rng()
        assert mock_object.call_count == 1
        assert original_seed != sampler._rng.seed
