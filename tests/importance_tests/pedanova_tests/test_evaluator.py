from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import numpy as np
import pytest

import optuna
from optuna.importance import PedAnovaImportanceEvaluator
from optuna.importance._ped_anova.evaluator import _QuantileFilter
from optuna.trial import FrozenTrial
from tests.importance_tests.test_importance_evaluators import get_study


_VALUES = list([[float(i)] for i in range(10)])[::-1]
_MULTI_VALUES = [[float(i), float(j)] for i, j in zip(range(10), reversed(range(10)))]


@pytest.mark.parametrize(
    "quantile,is_lower_better,values,target,filtered_indices",
    [
        (0.1, True, [[1.0], [2.0]], None, [0, 1]),  # Check min_n_trials = 2
        (0.49, True, deepcopy(_VALUES), None, list(range(10))[-5:]),
        (0.5, True, deepcopy(_VALUES), None, list(range(10))[-5:]),
        (0.51, True, deepcopy(_VALUES), None, list(range(10))[-6:]),
        (1.0, True, [[1.0], [2.0]], None, [0, 1]),
        (0.49, False, deepcopy(_VALUES), None, list(range(10))[:5]),
        (0.5, False, deepcopy(_VALUES), None, list(range(10))[:5]),
        (0.51, False, deepcopy(_VALUES), None, list(range(10))[:6]),
        # No tests for target!=None and is_lower_better=False because it is not used.
        (0.49, True, deepcopy(_MULTI_VALUES), lambda t: t.values[0], list(range(10))[:5]),
        (0.5, True, deepcopy(_MULTI_VALUES), lambda t: t.values[0], list(range(10))[:5]),
        (0.51, True, deepcopy(_MULTI_VALUES), lambda t: t.values[0], list(range(10))[:6]),
        (0.49, True, deepcopy(_MULTI_VALUES), lambda t: t.values[1], list(range(10))[-5:]),
        (0.5, True, deepcopy(_MULTI_VALUES), lambda t: t.values[1], list(range(10))[-5:]),
        (0.51, True, deepcopy(_MULTI_VALUES), lambda t: t.values[1], list(range(10))[-6:]),
    ],
)
def test_filter(
    quantile: float,
    is_lower_better: bool,
    values: list[list[float]],
    target: Callable[[FrozenTrial], float] | None,
    filtered_indices: list[int],
) -> None:
    _filter = _QuantileFilter(quantile, is_lower_better, min_n_top_trials=2, target=target)
    trials = [optuna.create_trial(values=vs) for vs in values]
    for i, t in enumerate(trials):
        t.set_user_attr("index", i)

    indices = [t.user_attrs["index"] for t in _filter.filter(trials)]
    assert len(indices) == len(filtered_indices)
    assert all(i == j for i, j in zip(indices, filtered_indices))


def test_error_in_ped_anova() -> None:
    with pytest.raises(RuntimeError):
        evaluator = PedAnovaImportanceEvaluator()
        study = get_study(seed=0, n_trials=5, is_multi_obj=True)
        evaluator.evaluate(study)


def test_n_trials_equal_to_min_n_top_trials() -> None:
    evaluator = PedAnovaImportanceEvaluator()
    study = get_study(seed=0, n_trials=evaluator._min_n_top_trials, is_multi_obj=False)
    param_importance = list(evaluator.evaluate(study).values())
    n_params = len(param_importance)
    assert np.allclose(param_importance, np.zeros(n_params))


def test_baseline_quantile_is_1() -> None:
    study = get_study(seed=0, n_trials=100, is_multi_obj=False)
    # baseline_quantile=1.0 enforces top_trials == all_trials identical.
    evaluator = PedAnovaImportanceEvaluator(baseline_quantile=1.0)
    param_importance = list(evaluator.evaluate(study).values())
    n_params = len(param_importance)
    # When top_trials == all_trials, all the importances become identical.
    assert np.allclose(param_importance, np.zeros(n_params))


def test_direction() -> None:
    study_minimize = get_study(seed=0, n_trials=20, is_multi_obj=False)
    study_maximize = optuna.create_study(direction="maximize")
    study_maximize.add_trials(study_minimize.trials)

    evaluator = PedAnovaImportanceEvaluator()
    assert evaluator.evaluate(study_minimize) != evaluator.evaluate(study_maximize)


def test_baseline_quantile() -> None:
    study = get_study(seed=0, n_trials=20, is_multi_obj=False)
    default_evaluator = PedAnovaImportanceEvaluator(baseline_quantile=0.1)
    evaluator = PedAnovaImportanceEvaluator(baseline_quantile=0.3)
    assert evaluator.evaluate(study) != default_evaluator.evaluate(study)


def test_evaluate_on_local() -> None:
    study = get_study(seed=0, n_trials=20, is_multi_obj=False)
    default_evaluator = PedAnovaImportanceEvaluator(evaluate_on_local=True)
    global_evaluator = PedAnovaImportanceEvaluator(evaluate_on_local=False)
    assert global_evaluator.evaluate(study) != default_evaluator.evaluate(study)
