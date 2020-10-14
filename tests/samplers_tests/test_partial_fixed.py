from typing import Union

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

    study_0 = optuna.create_study()
    study_0.sampler = RandomSampler(seed=42)
    study_0.optimize(objective, n_trials=1)
    x_sampled_0 = study_0.trials[0].params["x"]

    # Fix parameter y as 0.0.
    study_1 = optuna.create_study()
    study_1.sampler = PartialFixedSampler(
        fixed_params={"y": 0.0}, base_sampler=RandomSampler(seed=42)
    )
    study_1.optimize(objective, n_trials=1)
    x_sampled_1 = study_1.trials[0].params["x"]
    y_sampled_1 = study_1.trials[0].params["y"]
    assert x_sampled_1 == x_sampled_0
    assert y_sampled_1 == 0

    # Fix parameter y as the -5.
    study_2 = optuna.create_study()
    study_2.sampler = PartialFixedSampler(
        fixed_params={"y": -5}, base_sampler=RandomSampler(seed=42)
    )
    study_2.optimize(objective, n_trials=1)
    x_sampled_2 = study_2.trials[0].params["x"]
    y_sampled_2 = study_2.trials[0].params["y"]
    assert x_sampled_2 == x_sampled_0
    assert y_sampled_2 == -5


@pytest.mark.parametrize("fixed_y", [0.5, 5, 5.5])
def test_incompatible_types(fixed_y: Union[float, int]) -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_int("x", -10, 10)
        y = trial.suggest_int("y", -10, 10)
        return x ** 2 + y ** 2

    # Parameters of Int-type-distribution are rounded to int-type,
    # even if they are specified as float-type.
    study = optuna.create_study()
    study.sampler = PartialFixedSampler(fixed_params={"y": fixed_y}, base_sampler=study.sampler)
    study.optimize(objective, n_trials=1)
    assert study.trials[0].params["y"] == int(fixed_y)


@pytest.mark.parametrize("fixed_y", [-2, 2])
def test_out_of_the_range(fixed_y: int):
    def objective(trial: Trial) -> float:
        x = trial.suggest_int("x", -1, 1)
        y = trial.suggest_int("y", -1, 1)
        return x ** 2 + y ** 2

    # It is possible to fix parameters as out-of-the-range value.
    # Userwarnings will occur.
    study = optuna.create_study()
    study.sampler = PartialFixedSampler(fixed_params={"y": fixed_y}, base_sampler=study.sampler)
    with pytest.warns(UserWarning):
        study.optimize(objective, n_trials=1)
    assert study.trials[0].params["y"] == fixed_y
