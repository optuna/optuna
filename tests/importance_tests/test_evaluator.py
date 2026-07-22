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
from optuna.testing.pytest_importance import BasicImportanceEvaluatorTestCase
from optuna.testing.pytest_importance import ConditionalImportanceEvaluatorTestCase
from optuna.testing.pytest_importance import MultiObjectiveImportanceEvaluatorTestCase
from optuna.testing.pytest_importance import NonConditionalImportanceEvaluatorTestCase
from optuna.testing.pytest_importance import TreeBasedImportanceEvaluatorTestCase
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
    @pytest.fixture(params=ALL_EVALUATORS[2:])
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


class TestTreeBasedImportanceEvaluator(TreeBasedImportanceEvaluatorTestCase):
    @pytest.fixture(params=TREE_BASED_EVALUATORS)
    def evaluator(self, request: SubRequest) -> Callable[..., BaseImportanceEvaluator]:
        return request.param


def test_get_param_importances_unnormalized_experimental() -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        return x1**2

    study = create_study()
    study.optimize(objective, n_trials=4)
    with pytest.warns(ExperimentalWarning):
        get_param_importances(study, normalize=False)
