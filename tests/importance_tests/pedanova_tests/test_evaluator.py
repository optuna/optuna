from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import numpy as np
import pytest

import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.importance import PedAnovaImportanceEvaluator
from optuna.importance._ped_anova.evaluator import _QuantileFilter
from optuna.trial import FrozenTrial
from tests.importance_tests.test_importance_evaluators import get_study


_VALUES = list([[float(i)] for i in range(10)])[::-1]
_MULTI_VALUES = [[float(i), float(j)] for i, j in zip(range(10), reversed(range(10)))]


@pytest.mark.parametrize(
    "quantile,is_lower_better,values,target,filtered_indices",
    [
        (0.1, True, [[1.0], [2.0]], None, [0]),
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
    _filter = _QuantileFilter(quantile, is_lower_better, target=target)
    trials = [optuna.create_trial(values=vs) for vs in values]
    for i, t in enumerate(trials):
        t.set_user_attr("index", i)

    indices = [t.user_attrs["index"] for t in _filter.filter(trials)]
    assert len(indices) == len(filtered_indices)
    assert all(i == j for i, j in zip(indices, filtered_indices))


@pytest.mark.parametrize("n_trials", [0, 1, 2])
def test_n_trials_less_than_two(n_trials: int) -> None:
    evaluator = PedAnovaImportanceEvaluator()
    study = get_study(seed=0, n_trials=n_trials, is_multi_obj=False)
    param_importance = list(evaluator.evaluate(study).values())
    n_params = len(param_importance)
    if n_trials < 2:
        assert np.allclose(param_importance, np.zeros(n_params))
    else:
        assert not np.allclose(param_importance, np.zeros(n_params))


def test_direction() -> None:
    study_minimize = get_study(seed=0, n_trials=20, is_multi_obj=False)
    study_maximize = optuna.create_study(direction="maximize")
    study_maximize.add_trials(study_minimize.trials)

    evaluator = PedAnovaImportanceEvaluator()
    assert evaluator.evaluate(study_minimize) != evaluator.evaluate(study_maximize)


def test_target_quantile() -> None:
    study = get_study(seed=0, n_trials=20, is_multi_obj=False)
    default_evaluator = PedAnovaImportanceEvaluator(target_quantile=0.1)
    evaluator = PedAnovaImportanceEvaluator(target_quantile=0.3)
    assert evaluator.evaluate(study) != default_evaluator.evaluate(study)


def test_region_quantile_less_than_one() -> None:
    study = get_study(seed=0, n_trials=20, is_multi_obj=False)
    default_evaluator = PedAnovaImportanceEvaluator(region_quantile=1.0)
    evaluator = PedAnovaImportanceEvaluator(region_quantile=0.5)
    assert evaluator.evaluate(study) != default_evaluator.evaluate(study)


def test_evaluate_on_local() -> None:
    study = get_study(seed=0, n_trials=20, is_multi_obj=False)
    default_evaluator = PedAnovaImportanceEvaluator(evaluate_on_local=True)
    global_evaluator = PedAnovaImportanceEvaluator(evaluate_on_local=False)
    assert global_evaluator.evaluate(study) != default_evaluator.evaluate(study)


@pytest.mark.parametrize(
    "params", [None, [], ["c"], ["x"], ["c", "x"], ["x", "y"], ["c", "x", "y"], ["d"], ["c", "d"]]
)
def test_conditional(params: list[str] | None) -> None:
    study = optuna.study.create_study()
    dists_cx: dict[str, BaseDistribution] = {
        "c": FloatDistribution(0.0, 1.0),
        "x": FloatDistribution(-2.0, 0.0),
    }
    dists_cy: dict[str, BaseDistribution] = {
        "c": FloatDistribution(0.0, 1.0),
        "y": FloatDistribution(0.0, 2.0),
    }
    trials = [
        optuna.create_trial(params={"c": 1.0, "x": -1.0}, distributions=dists_cx, value=-1.0),
        optuna.create_trial(params={"c": 0.0, "y": 1.0}, distributions=dists_cy, value=1.0),
        optuna.create_trial(params={"c": 0.8, "x": -0.8}, distributions=dists_cx, value=-0.8),
        optuna.create_trial(params={"c": 0.2, "y": 0.2}, distributions=dists_cy, value=0.2),
        optuna.create_trial(params={"c": 0.8, "x": -0.6}, distributions=dists_cx, value=-0.6),
        optuna.create_trial(params={"c": 0.2, "y": 0.3}, distributions=dists_cy, value=0.3),
    ]
    study.add_trials(trials)
    evaluator = PedAnovaImportanceEvaluator()
    if params and "d" in params:
        with pytest.raises(ValueError):
            evaluator.evaluate(study, params=params)
        return
    importance = evaluator.evaluate(study, params=params)
    if params == []:
        assert importance == {}
        return
    assert set(importance.keys()) == set(params or ["c", "x", "y"])
    assert not all(v == 0.0 for v in importance.values()), f"{importance=}"


@pytest.mark.parametrize(
    (
        "directions",
        "values",
        "target_quantile",
        "region_quantile",
        "expected_target_size",
        "expected_region_size",
    ),
    [
        pytest.param(
            ["minimize", "minimize"],
            [[float(i), float(i)] for i in range(6)],
            0.5,
            0.8,
            3,
            5,
            id="different-nondomination-ranks",
        ),
        pytest.param(
            ["minimize", "minimize"],
            [
                [0.0, 0.0],
                [1.0, 5.0],
                [2.0, 4.0],
                [3.0, 3.0],
                [4.0, 2.0],
                [5.0, 1.0],
                [6.0, 6.0],
                [7.0, 7.0],
            ],
            0.5,
            0.75,
            4,
            6,
            id="same-rank-hssp-tie-break-after-best-rank",
        ),
        pytest.param(
            ["minimize", "minimize"],
            [[float(i), float(5 - i)] for i in range(6)],
            0.5,
            0.8,
            3,
            5,
            id="same-rank-hssp-tie-break-on-front",
        ),
    ],
)
def test_get_top_quantile_trials_multi_objective_target_none(
    directions: list[str],
    values: list[list[float]],
    target_quantile: float,
    region_quantile: float,
    expected_target_size: int,
    expected_region_size: int,
) -> None:
    study = optuna.create_study(directions=directions)
    study.add_trials([optuna.create_trial(values=vs) for vs in values])
    trials = study.get_trials(deepcopy=False)
    evaluator = PedAnovaImportanceEvaluator(
        target_quantile=target_quantile, region_quantile=region_quantile
    )

    target_trials = evaluator._get_top_quantile_trials(study, trials, target_quantile, target=None)
    region_trials = evaluator._get_top_quantile_trials(study, trials, region_quantile, target=None)

    assert len(target_trials) == expected_target_size
    assert len(region_trials) == expected_region_size
    assert {t._trial_id for t in target_trials}.issubset({t._trial_id for t in region_trials})
