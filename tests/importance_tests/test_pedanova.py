from __future__ import annotations

from collections.abc import Callable

import pytest

from optuna.importance import PedAnovaImportanceEvaluator
from optuna.study import create_study
from optuna.trial import FrozenTrial
from tests.importance_tests.common_tests import _test_evaluator_with_infinite
from tests.importance_tests.common_tests import get_study


def test_direction_of_ped_anova() -> None:
    study_minimize = get_study(seed=0, n_trials=20, is_multi_obj=False)
    study_maximize = create_study(direction="maximize")
    study_maximize.add_trials(study_minimize.trials)

    evaluator = PedAnovaImportanceEvaluator()
    assert evaluator.evaluate(study_minimize) != evaluator.evaluate(study_maximize)


def test_baseline_quantile_of_ped_anova() -> None:
    study = get_study(seed=0, n_trials=20, is_multi_obj=False)
    default_evaluator = PedAnovaImportanceEvaluator()
    evaluator = PedAnovaImportanceEvaluator(baseline_quantile=0.3)
    assert evaluator.evaluate(study) != default_evaluator.evaluate(study)


def test_evaluate_on_local_of_ped_anova() -> None:
    study = get_study(seed=0, n_trials=20, is_multi_obj=False)
    default_evaluator = PedAnovaImportanceEvaluator()
    global_evaluator = PedAnovaImportanceEvaluator(evaluate_on_local=False)
    assert global_evaluator.evaluate(study) != default_evaluator.evaluate(study)


@pytest.mark.parametrize(
    "custom_filter",
    [
        lambda trials: [t for t in trials if t.value < 5.0],
        lambda trials: [t for t, keep in zip(trials, [True, False] * 10) if keep],
    ],
)
def test_custom_filter_of_ped_anova(
    custom_filter: Callable[[list[FrozenTrial]], list[FrozenTrial]]
) -> None:
    study = get_study(seed=0, n_trials=20, is_multi_obj=False)
    default_evaluator = PedAnovaImportanceEvaluator()
    evaluator = PedAnovaImportanceEvaluator(custom_filter=custom_filter)
    assert evaluator.evaluate(study) != default_evaluator.evaluate(study)


@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
def test_pedanova_importance_evaluator_with_infinite(inf_value: float) -> None:
    _test_evaluator_with_infinite(PedAnovaImportanceEvaluator, inf_value)


@pytest.mark.parametrize("target_idx", [0, 1])
@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
def test_multi_objective_pedanova_importance_evaluator_with_infinite(
    target_idx: int, inf_value: float
) -> None:
    _test_evaluator_with_infinite(PedAnovaImportanceEvaluator, inf_value, target_idx)
