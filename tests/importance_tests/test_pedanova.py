import pytest

from optuna.importance import PedAnovaImportanceEvaluator
from tests.importance_tests.common_tests import _test_evaluator_with_infinite


def test_baseline_quantile_of_ped_anova():
    pass


def test_evaluate_on_local_of_ped_anova():
    pass


def test_custom_filter_of_ped_anova():
    pass


@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
def test_pedanova_importance_evaluator_with_infinite(inf_value: float) -> None:
    _test_evaluator_with_infinite(PedAnovaImportanceEvaluator, inf_value)


@pytest.mark.parametrize("target_idx", [0, 1])
@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
def test_multi_objective_pedanova_importance_evaluator_with_infinite(
    target_idx: int, inf_value: float
) -> None:
    _test_evaluator_with_infinite(PedAnovaImportanceEvaluator, inf_value, target_idx)
