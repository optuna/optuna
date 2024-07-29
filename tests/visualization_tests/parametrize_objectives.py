from __future__ import annotations

import pytest

import optuna


def objective_single_dynamic_with_categorical(trial: optuna.Trial) -> float:
    category = trial.suggest_categorical("category", ["foo", "bar"])
    if category == "foo":
        return (trial.suggest_float("x1", 0, 10) - 2) ** 2
    else:
        return -((trial.suggest_float("x2", -10, 0) + 5) ** 2)


def objective_single_none_categorical(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -100, 100)
    trial.suggest_categorical("y", ["foo", None])
    return x**2


parametrize_single_objective_functions = pytest.mark.parametrize(
    "objective_func",
    [
        objective_single_dynamic_with_categorical,
        objective_single_none_categorical,
    ],
)
