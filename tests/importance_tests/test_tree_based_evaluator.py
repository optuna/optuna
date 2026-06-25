from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from optuna import samplers
from optuna.importance import BaseImportanceEvaluator
from optuna.importance import FanovaImportanceEvaluator
from optuna.importance import get_param_importances
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
from optuna.samplers import RandomSampler
from optuna.study import create_study
from optuna.trial import Trial
from tests.importance_tests.test_evaluator import get_study


# Unlike PED-ANOVA, the tree-based evaluators only assess parameters that are present in all of the
# completed trials, reject dynamic search spaces, and are randomized (i.e. take a ``seed``). The
# tests in this file rely on these properties, so they are parametrized only over the tree-based
# evaluators. The PED-ANOVA counterparts are covered by ``pedanova_tests``.
tree_based_evaluators: list[type[BaseImportanceEvaluator]] = [
    MeanDecreaseImpurityImportanceEvaluator,
    FanovaImportanceEvaluator,
]

parametrize_tree_based_evaluator_cls = pytest.mark.parametrize(
    "evaluator_cls", tree_based_evaluators
)


@parametrize_tree_based_evaluator_cls
@pytest.mark.parametrize("normalize", [True, False])
def test_get_param_importances(
    evaluator_cls: Callable[[], BaseImportanceEvaluator], normalize: bool
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

    study = create_study(sampler=samplers.RandomSampler())
    study.optimize(objective, n_trials=3)

    param_importance = get_param_importances(study, evaluator=evaluator_cls(), normalize=normalize)

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


@parametrize_tree_based_evaluator_cls
@pytest.mark.parametrize("normalize", [True, False])
def test_get_param_importances_with_target(
    evaluator_cls: Callable[[], BaseImportanceEvaluator], normalize: bool
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

    study = create_study()
    study.optimize(objective, n_trials=3)

    param_importance = get_param_importances(
        study,
        evaluator=evaluator_cls(),
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


@parametrize_tree_based_evaluator_cls
def test_get_param_importances_invalid_dynamic_search_space_params(
    evaluator_cls: Callable[[], BaseImportanceEvaluator],
) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, trial.number + 0.1)
        return x1**2

    study = create_study()
    study.optimize(objective, n_trials=3)

    with pytest.raises(ValueError):
        get_param_importances(study, evaluator=evaluator_cls(), params=["x1"])


@parametrize_tree_based_evaluator_cls
def test_importance_evaluator_seed(evaluator_cls: Any) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        x2 = trial.suggest_float("x2", 0.1, 3, log=True)
        x3 = trial.suggest_float("x3", 2, 4, log=True)
        return x1 + x2 * x3

    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = evaluator_cls(seed=2)
    param_importance = evaluator.evaluate(study)

    evaluator = evaluator_cls(seed=2)
    param_importance_same_seed = evaluator.evaluate(study)
    assert param_importance == param_importance_same_seed

    evaluator = evaluator_cls(seed=3)
    param_importance_different_seed = evaluator.evaluate(study)
    assert param_importance != param_importance_different_seed


@parametrize_tree_based_evaluator_cls
def test_importance_evaluator_with_target(evaluator_cls: Any) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        x2 = trial.suggest_float("x2", 0.1, 3, log=True)
        x3 = trial.suggest_float("x3", 2, 4, log=True)
        return x1 + x2 * x3

    # Assumes that `seed` can be fixed to reproduce identical results.
    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = evaluator_cls(seed=0)
    param_importance = evaluator.evaluate(study)
    param_importance_with_target = evaluator.evaluate(
        study,
        target=lambda t: t.params["x1"] + t.params["x2"],
    )

    assert param_importance != param_importance_with_target


@parametrize_tree_based_evaluator_cls
def test_n_trees_of_tree_based_evaluator(
    evaluator_cls: type[FanovaImportanceEvaluator | MeanDecreaseImpurityImportanceEvaluator],
) -> None:
    study = get_study(seed=0, n_trials=3, is_multi_obj=False)
    evaluator = evaluator_cls(n_trees=10, seed=0)
    param_importance = evaluator.evaluate(study)

    evaluator = evaluator_cls(n_trees=20, seed=0)
    param_importance_different_n_trees = evaluator.evaluate(study)

    assert param_importance != param_importance_different_n_trees


@parametrize_tree_based_evaluator_cls
def test_max_depth_of_tree_based_evaluator(
    evaluator_cls: type[FanovaImportanceEvaluator | MeanDecreaseImpurityImportanceEvaluator],
) -> None:
    study = get_study(seed=0, n_trials=3, is_multi_obj=False)
    evaluator = evaluator_cls(max_depth=1, seed=0)
    param_importance = evaluator.evaluate(study)

    evaluator = evaluator_cls(max_depth=2, seed=0)
    param_importance_different_max_depth = evaluator.evaluate(study)

    assert param_importance != param_importance_different_max_depth
