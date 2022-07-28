from typing import Any
from typing import Dict
import warnings

import pytest

from optuna import create_study
from optuna.study import MaxTrialsCallback
from optuna.testing.objectives import pruned_objective
from optuna.trial import Trial
from optuna.trial import TrialState


def test_stop_with_MaxTrialsCallback() -> None:
    # Test stopping the optimization with MaxTrialsCallback.
    study = create_study()
    study.optimize(lambda _: 1.0, n_trials=10, callbacks=[MaxTrialsCallback(5)])
    assert len(study.trials) == 5

    # Test stopping the optimization with MaxTrialsCallback with pruned trials
    study = create_study()
    study.optimize(
        pruned_objective,
        n_trials=10,
        callbacks=[MaxTrialsCallback(5, states=(TrialState.PRUNED,))],
    )
    assert len(study.trials) == 5


def objective_for_warn_unused_parameter_callback_tests(trial: Trial) -> float:
    _ = trial.suggest_int("x0", 0, 10)
    _ = trial.suggest_float("x1", 0, 1.0)
    _ = trial.suggest_categorical("x2", ["foo", "bar"])
    return 1.0


@pytest.mark.parametrize(
    "param",
    [
        {"x0": 0, "x1": 0, "x2": "foo"},  # enqueue all valid parameters
        {"x0": 0},  # enqueue partial valid parameters
    ],
)
def test_warn_unused_parameter_callback_raise_no_warnings(param: Dict[str, Any]) -> None:
    study = create_study()
    study.enqueue_trial(param)
    with warnings.catch_warnings():
        study.optimize(objective_for_warn_unused_parameter_callback_tests, n_trials=1)


@pytest.mark.parametrize(
    "param",
    [
        {"x0": 0, "x1": 0, "x2": "foo", "x3": 0},  # x3 is not exist in search space
        {"x0": -1},  # x0 is invalid
    ],
)
def test_warn_unused_parameter_callback_raise_warnings(param: Dict[str, Any]) -> None:
    study = create_study()
    study.enqueue_trial(param)

    with warnings.catch_warnings():
        study.optimize(objective_for_warn_unused_parameter_callback_tests, n_trials=1)
