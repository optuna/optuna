import pytest

from optuna.importance import FanovaImportanceEvaluator
from tests.importance_tests.common_tests import _test_evaluator_with_infinite
from tests.importance_tests.common_tests import _test_max_depth_of_tree_based_evaluator
from tests.importance_tests.common_tests import _test_multi_objective_evaluator_with_infinite
from tests.importance_tests.common_tests import _test_n_trees_of_tree_based_evaluator


def test_fanova_importance_evaluator_n_trees() -> None:
    _test_n_trees_of_tree_based_evaluator(FanovaImportanceEvaluator)


def test_fanova_importance_evaluator_max_depth() -> None:
    _test_max_depth_of_tree_based_evaluator(FanovaImportanceEvaluator)


@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
def test_fanova_importance_evaluator_with_infinite(inf_value: float) -> None:
    _test_evaluator_with_infinite(FanovaImportanceEvaluator, inf_value)


@pytest.mark.parametrize("target_idx", [0, 1])
@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
def test_multi_objective_fanova_importance_evaluator_with_infinite(
    target_idx: int, inf_value: float
) -> None:
    _test_multi_objective_evaluator_with_infinite(
        FanovaImportanceEvaluator, target_idx=target_idx, inf_value=inf_value
    )
