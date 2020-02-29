from collections import OrderedDict
from typing import Callable
from typing import Type

import pytest

import optuna
from optuna.importance import BaseImportanceEvaluator
from optuna.importance import FanovaImportanceEvaluator
from optuna.importance import PermutationImportanceEvaluator
from optuna.importance import RandomForestFeatureImportanceEvaluator
from optuna.importance import get_param_importance
from optuna import samplers
from optuna import storages
from optuna.study import create_study
from optuna.trial import Trial

parametrize_storage = pytest.mark.parametrize(
    'storage_init_func',
    [storages.InMemoryStorage, lambda: storages.RDBStorage('sqlite:///:memory:')])


@parametrize_storage
@pytest.mark.parametrize(
    'evaluator_cls', [
        RandomForestFeatureImportanceEvaluator,
        PermutationImportanceEvaluator,
        FanovaImportanceEvaluator,
    ]
)
def test_get_param_importance(
        storage_init_func: Callable[[], storages.BaseStorage],
        evaluator_cls: Type[BaseImportanceEvaluator]) -> None:

    def objective(trial: Trial) -> float:
        x1 = trial.suggest_uniform('x1', 0.1, 3)
        x2 = trial.suggest_loguniform('x2', 0.1, 3)
        x3 = trial.suggest_discrete_uniform('x3', 0, 3, 1)
        x4 = trial.suggest_int('x4', -3, 3)
        x5 = trial.suggest_categorical('x5', [1.0, 1.1, 1.2])
        assert isinstance(x5, float)

        return x1 ** 4 + x2 + x3 - x4 ** 2 - x5

    study = create_study(storage_init_func(), sampler=samplers.RandomSampler())
    study.optimize(objective, n_trials=10)

    param_importance = get_param_importance(study, evaluator=evaluator_cls())

    assert isinstance(param_importance, OrderedDict)
    assert len(param_importance) == 5
    for param_name, importance in param_importance.items():
        assert isinstance(param_name, str)
        assert isinstance(importance, float)


def test_get_param_importance_empty_study() -> None:

    study = create_study()

    with pytest.raises(ValueError):
        get_param_importance(study)

    def objective(trial: Trial) -> float:
        raise optuna.exceptions.TrialPruned

    study.optimize(objective, n_trials=3)

    with pytest.raises(ValueError):
        get_param_importance(study)
