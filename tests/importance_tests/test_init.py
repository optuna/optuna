from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import optuna
from optuna import samplers
from optuna.exceptions import ExperimentalWarning
from optuna.importance import BaseImportanceEvaluator
from optuna.importance import FanovaImportanceEvaluator
from optuna.importance import get_param_importances
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
from optuna.samplers import RandomSampler
from optuna.study import create_study
from optuna.testing.objectives import pruned_objective
from optuna.testing.storages import STORAGE_MODES
from optuna.testing.storages import StorageSupplier
from optuna.trial import Trial


evaluators: list[type[BaseImportanceEvaluator]] = [
    MeanDecreaseImpurityImportanceEvaluator,
    FanovaImportanceEvaluator,
]

parametrize_evaluator = pytest.mark.parametrize("evaluator_init_func", evaluators)


@parametrize_evaluator
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_param_importance_target_is_none_and_study_is_multi_obj(
    storage_mode: str,
    evaluator_init_func: Callable[[], BaseImportanceEvaluator],
) -> None:
    def objective(trial: Trial) -> tuple[float, float]:
        x1 = trial.suggest_float("x1", 0.1, 3)
        x2 = trial.suggest_float("x2", 0.1, 3, log=True)
        x3 = trial.suggest_float("x3", 0, 3, step=1)
        x4 = trial.suggest_int("x4", -3, 3)
        x5 = trial.suggest_int("x5", 1, 5, log=True)
        x6 = trial.suggest_categorical("x6", [1.0, 1.1, 1.2])
        if trial.number % 2 == 0:
            # Conditional parameters are ignored unless `params` is specified and is not `None`.
            x7 = trial.suggest_float("x7", 0.1, 3)

        value = x1**4 + x2 + x3 - x4**2 - x5 + x6
        if trial.number % 2 == 0:
            value += x7
        return value, 0.0

    with StorageSupplier(storage_mode) as storage:
        study = create_study(directions=["minimize", "minimize"], storage=storage)
        study.optimize(objective, n_trials=3)

        with pytest.raises(ValueError):
            get_param_importances(study, evaluator=evaluator_init_func())


@parametrize_evaluator
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize("normalize", [True, False])
def test_get_param_importances(
    storage_mode: str, evaluator_init_func: Callable[[], BaseImportanceEvaluator], normalize: bool
) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        x2 = trial.suggest_float("x2", 0.1, 3, log=True)
        x3 = trial.suggest_float("x3", 0, 3, step=1)
        x4 = trial.suggest_int("x4", -3, 3)
        x5 = trial.suggest_int("x5", 1, 5, log=True)
        x6 = trial.suggest_categorical("x6", [1.0, 1.1, 1.2])
        if trial.number % 2 == 0:
            # Conditional parameters are ignored unless `params` is specified and is not `None`.
            x7 = trial.suggest_float("x7", 0.1, 3)

        value = x1**4 + x2 + x3 - x4**2 - x5 + x6
        if trial.number % 2 == 0:
            value += x7
        return value

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=samplers.RandomSampler())
        study.optimize(objective, n_trials=3)

        param_importance = get_param_importances(
            study, evaluator=evaluator_init_func(), normalize=normalize
        )

        assert isinstance(param_importance, dict)
        assert len(param_importance) == 6
        assert all(
            param_name in param_importance for param_name in ["x1", "x2", "x3", "x4", "x5", "x6"]
        )
        prev_importance = float("inf")
        for param_name, importance in param_importance.items():
            assert isinstance(param_name, str)
            assert isinstance(importance, float)
            assert importance <= prev_importance
            prev_importance = importance

        # Sanity check for param importances
        assert all(0 <= x < float("inf") for x in param_importance.values())
        if normalize:
            assert np.isclose(sum(param_importance.values()), 1.0)


@parametrize_evaluator
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize("params", [[], ["x1"], ["x1", "x3"], ["x1", "x4"]])
@pytest.mark.parametrize("normalize", [True, False])
def test_get_param_importances_with_params(
    storage_mode: str,
    params: list[str],
    evaluator_init_func: Callable[[], BaseImportanceEvaluator],
    normalize: bool,
) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        x2 = trial.suggest_float("x2", 0.1, 3, log=True)
        x3 = trial.suggest_float("x3", 0, 3, step=1)
        if trial.number % 2 == 0:
            x4 = trial.suggest_float("x4", 0.1, 3)

        value = x1**4 + x2 + x3
        if trial.number % 2 == 0:
            value += x4
        return value

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(objective, n_trials=10)

        param_importance = get_param_importances(
            study, evaluator=evaluator_init_func(), params=params, normalize=normalize
        )

        assert isinstance(param_importance, dict)
        assert len(param_importance) == len(params)
        assert all(param in param_importance for param in params)
        for param_name, importance in param_importance.items():
            assert isinstance(param_name, str)
            assert isinstance(importance, float)

        # Sanity check for param importances
        assert all(0 <= x < float("inf") for x in param_importance.values())
        if normalize:
            assert len(param_importance) == 0 or np.isclose(sum(param_importance.values()), 1.0)


def test_get_param_importances_unnormalized_experimental() -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        return x1**2

    study = create_study()
    study.optimize(objective, n_trials=4)
    with pytest.warns(ExperimentalWarning):
        get_param_importances(study, normalize=False)


@parametrize_evaluator
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize("normalize", [True, False])
def test_get_param_importances_with_target(
    storage_mode: str, evaluator_init_func: Callable[[], BaseImportanceEvaluator], normalize: bool
) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        x2 = trial.suggest_float("x2", 0.1, 3, log=True)
        x3 = trial.suggest_float("x3", 0, 3, step=1)
        if trial.number % 2 == 0:
            x4 = trial.suggest_float("x4", 0.1, 3)

        value = x1**4 + x2 + x3
        if trial.number % 2 == 0:
            value += x4
        return value

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(objective, n_trials=3)

        param_importance = get_param_importances(
            study,
            evaluator=evaluator_init_func(),
            target=lambda t: t.params["x1"] + t.params["x2"],
            normalize=normalize,
        )

        assert isinstance(param_importance, dict)
        assert len(param_importance) == 3
        assert all(param_name in param_importance for param_name in ["x1", "x2", "x3"])
        prev_importance = float("inf")
        for param_name, importance in param_importance.items():
            assert isinstance(param_name, str)
            assert isinstance(importance, float)
            assert importance <= prev_importance
            prev_importance = importance

        # Sanity check for param importances
        assert all(0 <= x < float("inf") for x in param_importance.values())
        if normalize:
            assert np.isclose(sum(param_importance.values()), 1.0)


@parametrize_evaluator
def test_get_param_importances_invalid_empty_study(
    evaluator_init_func: Callable[[], BaseImportanceEvaluator],
) -> None:
    study = create_study()

    with pytest.raises(ValueError):
        get_param_importances(study, evaluator=evaluator_init_func())

    study.optimize(pruned_objective, n_trials=3)

    with pytest.raises(ValueError):
        get_param_importances(study, evaluator=evaluator_init_func())


@parametrize_evaluator
def test_get_param_importances_invalid_single_trial(
    evaluator_init_func: Callable[[], BaseImportanceEvaluator],
) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        return x1**2

    study = create_study()
    study.optimize(objective, n_trials=1)

    with pytest.raises(ValueError):
        get_param_importances(study, evaluator=evaluator_init_func())


@parametrize_evaluator
def test_get_param_importances_invalid_no_completed_trials_params(
    evaluator_init_func: Callable[[], BaseImportanceEvaluator],
) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        if trial.number % 2 == 0:
            _ = trial.suggest_float("x2", 0.1, 3, log=True)
            raise optuna.TrialPruned
        return x1**2

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
    evaluator_init_func: Callable[[], BaseImportanceEvaluator],
) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, trial.number + 0.1)
        return x1**2

    study = create_study()
    study.optimize(objective, n_trials=3)

    with pytest.raises(ValueError):
        get_param_importances(study, evaluator=evaluator_init_func(), params=["x1"])


@parametrize_evaluator
def test_get_param_importances_empty_search_space(
    evaluator_init_func: Callable[[], BaseImportanceEvaluator],
) -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_float("x", 0, 5)
        y = trial.suggest_float("y", 1, 1)
        return 4 * x**2 + 4 * y**2

    study = create_study()
    study.optimize(objective, n_trials=3)

    param_importance = get_param_importances(study, evaluator=evaluator_init_func())

    assert len(param_importance) == 2
    assert all([param in param_importance for param in ["x", "y"]])
    assert param_importance["x"] > 0.0
    assert param_importance["y"] == 0.0


@parametrize_evaluator
def test_importance_evaluator_seed(evaluator_init_func: Any) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        x2 = trial.suggest_float("x2", 0.1, 3, log=True)
        x3 = trial.suggest_float("x3", 2, 4, log=True)
        return x1 + x2 * x3

    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = evaluator_init_func(seed=2)
    param_importance = evaluator.evaluate(study)

    evaluator = evaluator_init_func(seed=2)
    param_importance_same_seed = evaluator.evaluate(study)
    assert param_importance == param_importance_same_seed

    evaluator = evaluator_init_func(seed=3)
    param_importance_different_seed = evaluator.evaluate(study)
    assert param_importance != param_importance_different_seed


@parametrize_evaluator
def test_importance_evaluator_with_target(evaluator_init_func: Any) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        x2 = trial.suggest_float("x2", 0.1, 3, log=True)
        x3 = trial.suggest_float("x3", 2, 4, log=True)
        return x1 + x2 * x3

    # Assumes that `seed` can be fixed to reproduce identical results.
    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = evaluator_init_func(seed=0)
    param_importance = evaluator.evaluate(study)
    param_importance_with_target = evaluator.evaluate(
        study,
        target=lambda t: t.params["x1"] + t.params["x2"],
    )

    assert param_importance != param_importance_with_target
