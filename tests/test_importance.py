from collections import OrderedDict
import math
from typing import Callable
from typing import List

import pytest

import optuna
from optuna.importance import BaseImportanceEvaluator
from optuna.importance import FanovaImportanceEvaluator
from optuna.importance import get_param_importances
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
from optuna import samplers
from optuna import storages
from optuna.study import create_study
from optuna.trial import Trial

parametrize_storage = pytest.mark.parametrize(
    "storage_init_func",
    [storages.InMemoryStorage, lambda: storages.RDBStorage("sqlite:///:memory:")],
)

parametrize_evaluator = pytest.mark.parametrize(
    "evaluator_init_func", [MeanDecreaseImpurityImportanceEvaluator, FanovaImportanceEvaluator]
)


@parametrize_evaluator
@parametrize_storage
def test_get_param_importances(
    storage_init_func: Callable[[], storages.BaseStorage],
    evaluator_init_func: Callable[[], BaseImportanceEvaluator],
) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_uniform("x1", 0.1, 3)
        x2 = trial.suggest_loguniform("x2", 0.1, 3)
        x3 = trial.suggest_discrete_uniform("x3", 0, 3, 1)
        x4 = trial.suggest_int("x4", -3, 3)
        x5 = trial.suggest_categorical("x5", [1.0, 1.1, 1.2])
        if trial.number % 2 == 0:
            # Conditional parameters are ignored unless `params` is specified and is not `None`.
            x6 = trial.suggest_uniform("x6", 0.1, 3)

        assert isinstance(x5, float)
        value = x1 ** 4 + x2 + x3 - x4 ** 2 - x5
        if trial.number % 2 == 0:
            value += x6
        return value

    study = create_study(storage_init_func(), sampler=samplers.RandomSampler())
    study.optimize(objective, n_trials=3)

    param_importance = get_param_importances(study, evaluator=evaluator_init_func())

    assert isinstance(param_importance, OrderedDict)
    assert len(param_importance) == 5
    assert all(param_name in param_importance for param_name in ["x1", "x2", "x3", "x4", "x5"])
    prev_importance = float("inf")
    for param_name, importance in param_importance.items():
        assert isinstance(param_name, str)
        assert isinstance(importance, float)
        assert importance <= prev_importance
        prev_importance = importance
    assert math.isclose(1.0, sum(i for i in param_importance.values()), abs_tol=1e-5)


@parametrize_evaluator
@parametrize_storage
@pytest.mark.parametrize("params", [[], ["x1"], ["x1", "x3"], ["x1", "x4"]])
def test_get_param_importances_with_params(
    storage_init_func: Callable[[], storages.BaseStorage],
    params: List[str],
    evaluator_init_func: Callable[[], BaseImportanceEvaluator],
) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_uniform("x1", 0.1, 3)
        x2 = trial.suggest_loguniform("x2", 0.1, 3)
        x3 = trial.suggest_discrete_uniform("x3", 0, 3, 1)
        if trial.number % 2 == 0:
            x4 = trial.suggest_uniform("x4", 0.1, 3)

        value = x1 ** 4 + x2 + x3
        if trial.number % 2 == 0:
            value += x4
        return value

    study = create_study(storage_init_func())
    study.optimize(objective, n_trials=10)

    param_importance = get_param_importances(study, evaluator=evaluator_init_func(), params=params)

    assert isinstance(param_importance, OrderedDict)
    assert len(param_importance) == len(params)
    assert all(param in param_importance for param in params)
    for param_name, importance in param_importance.items():
        assert isinstance(param_name, str)
        assert isinstance(importance, float)
    if len(param_importance) > 0:
        assert math.isclose(1.0, sum(i for i in param_importance.values()), abs_tol=1e-5)


@parametrize_evaluator
def test_get_param_importances_invalid_empty_study(
    evaluator_init_func: Callable[[], BaseImportanceEvaluator]
) -> None:

    study = create_study()

    with pytest.raises(ValueError):
        get_param_importances(study, evaluator=evaluator_init_func())

    def objective(trial: Trial) -> float:
        raise optuna.TrialPruned

    study.optimize(objective, n_trials=3)

    with pytest.raises(ValueError):
        get_param_importances(study, evaluator=evaluator_init_func())


@parametrize_evaluator
def test_get_param_importances_invalid_single_trial(
    evaluator_init_func: Callable[[], BaseImportanceEvaluator]
) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_uniform("x1", 0.1, 3)
        return x1 ** 2

    study = create_study()
    study.optimize(objective, n_trials=1)

    with pytest.raises(ValueError):
        get_param_importances(study, evaluator=evaluator_init_func())


def test_get_param_importances_invalid_evaluator_type() -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_uniform("x1", 0.1, 3)
        return x1 ** 2

    study = create_study()
    study.optimize(objective, n_trials=3)

    with pytest.raises(TypeError):
        get_param_importances(study, evaluator={})


@parametrize_evaluator
def test_get_param_importances_invalid_no_completed_trials_params(
    evaluator_init_func: Callable[[], BaseImportanceEvaluator]
) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_uniform("x1", 0.1, 3)
        if trial.number % 2 == 0:
            _ = trial.suggest_loguniform("x2", 0.1, 3)
            raise optuna.TrialPruned
        return x1 ** 2

    study = create_study()
    study.optimize(objective, n_trials=3)

    # None of the trials with `x2` are completed.
    with pytest.raises(ValueError):
        get_param_importances(study, evaluator=evaluator_init_func(), params=["x2"])

    # None of the trials with `x2` are completed. Adding "x1" should not matter.
    with pytest.raises(ValueError):
        get_param_importances(study, evaluator=evaluator_init_func(), params=["x1", "x2"])

    # None of the trials contain `x3`.
    with pytest.raises(ValueError):
        get_param_importances(study, evaluator=evaluator_init_func(), params=["x3"])


@parametrize_evaluator
def test_get_param_importances_invalid_dynamic_search_space_params(
    evaluator_init_func: Callable[[], BaseImportanceEvaluator]
) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_uniform("x1", 0.1, trial.number + 0.1)
        return x1 ** 2

    study = create_study()
    study.optimize(objective, n_trials=3)

    with pytest.raises(ValueError):
        get_param_importances(study, evaluator=evaluator_init_func(), params=["x1"])


@parametrize_evaluator
def test_get_param_importances_invalid_params_type(
    evaluator_init_func: Callable[[], BaseImportanceEvaluator]
) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_uniform("x1", 0.1, 3)
        return x1 ** 2

    study = create_study()
    study.optimize(objective, n_trials=3)

    with pytest.raises(TypeError):
        get_param_importances(study, evaluator=evaluator_init_func(), params={})

    with pytest.raises(TypeError):
        get_param_importances(study, evaluator=evaluator_init_func(), params=[0])
