import pytest

import optuna
from optuna import samplers
from optuna.trial import Trial


def test_study_optimize_with_single_search_space() -> None:
    def objective(trial: Trial) -> float:
        a = trial.suggest_int("a", 0, 2)

        if a == 0:
            b = trial.suggest_float("b", -1.0, 1.0, step=0.5)
            return a + b
        elif a == 1:
            c = trial.suggest_categorical("c", ["x", "y", None])
            if c == "x":
                return a + 1
            else:
                return a - 1
        else:
            return a * 2

    study = optuna.create_study(sampler=samplers.BruteForceSampler())
    study.optimize(objective)

    expected_suggested_values = [
        {"a": 0, "b": -1.0},
        {"a": 0, "b": -0.5},
        {"a": 0, "b": 0.0},
        {"a": 0, "b": 0.5},
        {"a": 0, "b": 1.0},
        {"a": 1, "c": "x"},
        {"a": 1, "c": "y"},
        {"a": 1, "c": None},
        {"a": 2},
    ]
    all_suggested_values = [t.params for t in study.trials]
    assert len(all_suggested_values) == len(expected_suggested_values)
    for a in all_suggested_values:
        assert a in expected_suggested_values


def test_study_optimize_with_infinite_search_space() -> None:
    def objective(trial: Trial) -> float:
        return trial.suggest_float("a", 0, 2)

    study = optuna.create_study(sampler=samplers.BruteForceSampler())

    with pytest.raises(ValueError):
        study.optimize(objective)
