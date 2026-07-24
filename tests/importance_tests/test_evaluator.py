from __future__ import annotations

from collections.abc import Callable
from typing import Any

from _pytest.fixtures import SubRequest
import pytest

from optuna.exceptions import ExperimentalWarning
from optuna.importance import BaseImportanceEvaluator
from optuna.importance import FanovaImportanceEvaluator
from optuna.importance import get_param_importances
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
from optuna.importance import PedAnovaImportanceEvaluator
from optuna.study import create_study
from optuna.testing.pytest_importance import _get_study
from optuna.testing.pytest_importance import BasicImportanceEvaluatorTestCase
from optuna.testing.pytest_importance import ConditionalImportanceEvaluatorTestCase
from optuna.testing.pytest_importance import MultiObjectiveImportanceEvaluatorTestCase
from optuna.testing.pytest_importance import NonConditionalImportanceEvaluatorTestCase
from optuna.trial import Trial


def _mean_decrease_impurity_evaluator(
    seed: int | None = 0, **kwargs: Any
) -> BaseImportanceEvaluator:
    return MeanDecreaseImpurityImportanceEvaluator(seed=seed, **kwargs)


def _fanova_evaluator(seed: int | None = 0, **kwargs: Any) -> BaseImportanceEvaluator:
    return FanovaImportanceEvaluator(seed=seed, **kwargs)


def _ped_anova_evaluator(seed: int | None = None, **kwargs: Any) -> BaseImportanceEvaluator:
    return PedAnovaImportanceEvaluator()


ALL_EVALUATORS = [
    pytest.param(_mean_decrease_impurity_evaluator, id="MeanDecreaseImpurityImportanceEvaluator"),
    pytest.param(_fanova_evaluator, id="FanovaImportanceEvaluator"),
    pytest.param(_ped_anova_evaluator, id="PedAnovaImportanceEvaluator"),
]
TREE_BASED_EVALUATORS = ALL_EVALUATORS[:2]


class TestBasicImportanceEvaluator(BasicImportanceEvaluatorTestCase):
    @pytest.fixture(params=ALL_EVALUATORS)
    def evaluator(self, request: SubRequest) -> Callable[..., BaseImportanceEvaluator]:
        return request.param


class TestConditionalImportanceEvaluator(ConditionalImportanceEvaluatorTestCase):
    @pytest.fixture(params=[pytest.param(_ped_anova_evaluator, id="PedAnovaImportanceEvaluator")])
    def evaluator(self, request: SubRequest) -> Callable[..., BaseImportanceEvaluator]:
        return request.param


class TestNonConditionalImportanceEvaluator(NonConditionalImportanceEvaluatorTestCase):
    @pytest.fixture(params=TREE_BASED_EVALUATORS)
    def evaluator(self, request: SubRequest) -> Callable[..., BaseImportanceEvaluator]:
        return request.param


class TestMultiObjectiveImportanceEvaluator(MultiObjectiveImportanceEvaluatorTestCase):
    @pytest.fixture(params=ALL_EVALUATORS[2:])
    def evaluator(self, request: SubRequest) -> Callable[..., BaseImportanceEvaluator]:
        return request.param


@pytest.mark.parametrize("evaluator", TREE_BASED_EVALUATORS)
def test_n_trees_of_tree_based_evaluator(
    evaluator: Callable[..., BaseImportanceEvaluator],
) -> None:
    study = _get_study(seed=0, n_trials=3, is_multi_obj=False)
    param_importance = evaluator(n_trees=10, seed=0).evaluate(study)
    param_importance_different_n_trees = evaluator(n_trees=20, seed=0).evaluate(study)

    assert param_importance != param_importance_different_n_trees


@pytest.mark.parametrize("evaluator", TREE_BASED_EVALUATORS)
def test_max_depth_of_tree_based_evaluator(
    evaluator: Callable[..., BaseImportanceEvaluator],
) -> None:
    study = _get_study(seed=0, n_trials=3, is_multi_obj=False)
    param_importance = evaluator(max_depth=1, seed=0).evaluate(study)
    param_importance_different_max_depth = evaluator(max_depth=2, seed=0).evaluate(study)

    assert param_importance != param_importance_different_max_depth


@pytest.mark.parametrize("evaluator", TREE_BASED_EVALUATORS)
def test_importance_evaluator_seed(
    evaluator: Callable[..., BaseImportanceEvaluator],
) -> None:
    study = _get_study(seed=0, n_trials=3, is_multi_obj=False)

    param_importance = evaluator(seed=2).evaluate(study)
    param_importance_same_seed = evaluator(seed=2).evaluate(study)
    assert param_importance == param_importance_same_seed

    param_importance_different_seed = evaluator(seed=3).evaluate(study)
    assert param_importance != param_importance_different_seed


def test_get_param_importances_unnormalized_experimental() -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        return x1**2

    study = create_study()
    study.optimize(objective, n_trials=4)
    with pytest.warns(ExperimentalWarning):
        get_param_importances(study, normalize=False)
