from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import optuna
from optuna import samplers
from optuna import Study
from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.importance import BaseImportanceEvaluator
from optuna.importance import FanovaImportanceEvaluator
from optuna.importance import get_param_importances
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
from optuna.importance import PedAnovaImportanceEvaluator
from optuna.samplers import RandomSampler
from optuna.study import create_study
from optuna.testing.objectives import pruned_objective
from optuna.trial import create_trial
from optuna.trial import Trial


all_evaluators: list[Callable[..., BaseImportanceEvaluator]] = [
    lambda seed=0, **kwargs: MeanDecreaseImpurityImportanceEvaluator(seed=seed, **kwargs),
    lambda seed=0, **kwargs: FanovaImportanceEvaluator(seed=seed, **kwargs),
    lambda seed=None, **kwargs: PedAnovaImportanceEvaluator(),
]
non_conditional_evaluators: list[Callable[..., BaseImportanceEvaluator]] = [
    lambda seed=0, **kwargs: MeanDecreaseImpurityImportanceEvaluator(seed=seed, **kwargs),
    lambda seed=0, **kwargs: FanovaImportanceEvaluator(seed=seed, **kwargs),
]
conditional_supported_evaluators: list[Callable[..., BaseImportanceEvaluator]] = [
    lambda seed=None, **kwargs: PedAnovaImportanceEvaluator(),
]
single_objective_primary_evaluators: list[Callable[..., BaseImportanceEvaluator]] = [
    lambda seed=0, **kwargs: MeanDecreaseImpurityImportanceEvaluator(seed=seed, **kwargs),
    lambda seed=0, **kwargs: FanovaImportanceEvaluator(seed=seed, **kwargs),
]
multi_objective_supported_evaluators: list[Callable[..., BaseImportanceEvaluator]] = [
    lambda seed=None, **kwargs: PedAnovaImportanceEvaluator(),
]
tree_based_evaluators: list[Callable[..., BaseImportanceEvaluator]] = [
    lambda seed=0, **kwargs: MeanDecreaseImpurityImportanceEvaluator(seed=seed, **kwargs),
    lambda seed=0, **kwargs: FanovaImportanceEvaluator(seed=seed, **kwargs),
]

_evaluator_ids = [
    "MeanDecreaseImpurityImportanceEvaluator",
    "FanovaImportanceEvaluator",
    "PedAnovaImportanceEvaluator",
]

parametrize_all = pytest.mark.parametrize("evaluator_cls", all_evaluators, ids=_evaluator_ids)
parametrize_non_conditional = pytest.mark.parametrize(
    "evaluator_cls", non_conditional_evaluators, ids=_evaluator_ids[:2]
)
parametrize_conditional_supported = pytest.mark.parametrize(
    "evaluator_cls", conditional_supported_evaluators, ids=_evaluator_ids[2:]
)
parametrize_single_objective_primary = pytest.mark.parametrize(
    "evaluator_cls", single_objective_primary_evaluators, ids=_evaluator_ids[:2]
)
parametrize_multi_objective_supported = pytest.mark.parametrize(
    "evaluator_cls", multi_objective_supported_evaluators, ids=_evaluator_ids[2:]
)
parametrize_tree_based = pytest.mark.parametrize(
    "evaluator_cls", tree_based_evaluators, ids=_evaluator_ids[:2]
)


def objective(trial: Trial) -> float:
    x1 = trial.suggest_float("x1", 0.1, 3)
    x2 = trial.suggest_float("x2", 0.1, 3, log=True)
    x3 = trial.suggest_float("x3", 2, 4, log=True)
    return x1 + x2 * x3


def multi_objective_function(trial: Trial) -> tuple[float, float]:
    x1 = trial.suggest_float("x1", 0.1, 3)
    x2 = trial.suggest_float("x2", 0.1, 3, log=True)
    x3 = trial.suggest_float("x3", 2, 4, log=True)
    return x1, x2 * x3


def get_study(seed: int, n_trials: int, is_multi_obj: bool) -> Study:
    # Assumes that `seed` can be fixed to reproduce identical results.
    directions = ["minimize", "minimize"] if is_multi_obj else ["minimize"]
    study = create_study(sampler=RandomSampler(seed=seed), directions=directions)
    if is_multi_obj:
        study.optimize(multi_objective_function, n_trials=n_trials)
    else:
        study.optimize(objective, n_trials=n_trials)

    return study


# Tests for all evaluators - common behavior


def test_get_param_importances_unnormalized_experimental() -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        return x1**2

    study = create_study()
    study.optimize(objective, n_trials=4)
    with pytest.warns(ExperimentalWarning):
        get_param_importances(study, normalize=False)


@parametrize_all
def test_get_param_importances_invalid_empty_study(
    evaluator_cls: Callable[[], BaseImportanceEvaluator],
) -> None:
    study = create_study()

    importance = get_param_importances(study, evaluator=evaluator_cls())
    assert isinstance(importance, dict)
    assert not importance

    study.optimize(pruned_objective, n_trials=3)

    importance = get_param_importances(study, evaluator=evaluator_cls())
    assert isinstance(importance, dict)
    assert not importance


@parametrize_all
def test_get_param_importances_invalid_single_trial(
    evaluator_cls: Callable[[], BaseImportanceEvaluator],
) -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        return x1**2

    study = create_study()
    study.optimize(objective, n_trials=1)

    importance = get_param_importances(study, evaluator=evaluator_cls())
    assert importance == {"x1": 1.0}  # becomes 1.0 after normalization


@parametrize_all
def test_get_param_importances_invalid_no_completed_trials_params(
    evaluator_cls: Callable[[], BaseImportanceEvaluator],
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
        get_param_importances(study, evaluator=evaluator_cls(), params=["x2"])

    # None of the trials with `x2` are completed. Adding "x1" should not matter.
    with pytest.raises(ValueError):
        get_param_importances(study, evaluator=evaluator_cls(), params=["x1", "x2"])

    # None of the trials contain `x3`.
    with pytest.raises(ValueError):
        get_param_importances(study, evaluator=evaluator_cls(), params=["x3"])


@parametrize_all
def test_get_param_importances_empty_search_space(
    evaluator_cls: Callable[[], BaseImportanceEvaluator],
) -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_float("x", 0, 5)
        y = trial.suggest_float("y", 1, 1)
        return 4 * x**2 + 4 * y**2

    study = create_study()
    study.optimize(objective, n_trials=3)

    param_importance = get_param_importances(study, evaluator=evaluator_cls())

    assert len(param_importance) == 2
    assert all([param in param_importance for param in ["x", "y"]])
    assert param_importance["x"] > 0.0
    assert param_importance["y"] == 0.0


@pytest.mark.filterwarnings("ignore::UserWarning")
@parametrize_all
@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
@pytest.mark.parametrize("target_idx", [0, 1, None])
def test_evaluator_with_infinite(
    evaluator_cls: Callable[[], BaseImportanceEvaluator],
    inf_value: float,
    target_idx: int | None,
) -> None:
    # The test ensures that trials with infinite values are ignored to calculate importance scores.
    evaluator = evaluator_cls()
    is_multi_obj = target_idx is not None
    study = get_study(seed=13, n_trials=10, is_multi_obj=is_multi_obj)
    target = (lambda t: t.values[target_idx]) if is_multi_obj else None  # noqa: E731
    # Importance scores are calculated without a trial with an inf value.
    param_importance_without_inf = evaluator.evaluate(study, target=target)

    # A trial with an inf value is added into the study manually.
    study.add_trial(
        create_trial(
            values=[inf_value] if not is_multi_obj else [inf_value, inf_value],
            params={"x1": 1.0, "x2": 1.0, "x3": 3.0},
            distributions={
                "x1": FloatDistribution(low=0.1, high=3),
                "x2": FloatDistribution(low=0.1, high=3, log=True),
                "x3": FloatDistribution(low=2, high=4, log=True),
            },
        )
    )
    # Importance scores are calculated with a trial with an inf value.
    param_importance_with_inf = evaluator.evaluate(study, target=target)

    # Obtained importance scores should be the same between with inf and without inf,
    # because the last trial whose objective value is an inf is ignored.
    # PED-ANOVA can handle inf, so anyways the length should be identical.
    assert param_importance_with_inf == param_importance_without_inf


@parametrize_all
def test_evaluator_with_only_single_dists(
    evaluator_cls: Callable[[], BaseImportanceEvaluator],
) -> None:
    evaluator = evaluator_cls()
    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(lambda trial: trial.suggest_float("a", 0.0, 0.0), n_trials=3)
    param_importance = evaluator.evaluate(study)

    assert param_importance == {"a": 0.0}


@parametrize_all
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
        target=lambda t: t.params["x3"],
    )

    assert param_importance != param_importance_with_target


# Tests for conditional parameter handling


@parametrize_conditional_supported
@pytest.mark.parametrize("params", [[], ["x1"], ["x1", "x3"], ["x1", "x4"]])
@pytest.mark.parametrize("normalize", [True, False])
def test_get_param_importances_with_params(
    params: list[str],
    evaluator_cls: Callable[[], BaseImportanceEvaluator],
    normalize: bool,
) -> None:
    """Test that conditional parameters are properly handled by supported evaluators."""

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
    study.optimize(objective, n_trials=10)

    param_importance = get_param_importances(
        study, evaluator=evaluator_cls(), params=params, normalize=normalize
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


@parametrize_conditional_supported
@pytest.mark.parametrize(
    "params", [None, [], ["c"], ["x"], ["c", "x"], ["x", "y"], ["c", "x", "y"], ["d"], ["c", "d"]]
)
def test_conditional_parameters(
    evaluator_cls: Callable[[], BaseImportanceEvaluator],
    params: list[str] | None,
) -> None:
    """Test that conditional parameters are handled correctly by supported evaluators."""
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
    evaluator = evaluator_cls()
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


@parametrize_non_conditional
@pytest.mark.parametrize("normalize", [True, False])
def test_get_param_importances_non_conditional(
    evaluator_cls: Callable[[], BaseImportanceEvaluator], normalize: bool
) -> None:
    """Test that tree-based evaluators ignore conditional parameters."""

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


# Tests for multi-objective support


@parametrize_multi_objective_supported
def test_get_param_importance_target_is_none_and_study_is_multi_obj(
    evaluator_cls: Callable[[], BaseImportanceEvaluator],
) -> None:
    """Test that evaluators supporting multi-objective work without target specification."""

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

    study = create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=3)

    # Evaluators in multi_objective_supported_evaluators must support multi-objective
    # without requiring a target specification
    param_importance = get_param_importances(study, evaluator=evaluator_cls())
    assert isinstance(param_importance, dict)


# Tests for single-objective primary evaluators (tree-based)


@parametrize_single_objective_primary
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


@parametrize_single_objective_primary
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


@parametrize_single_objective_primary
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


# Tests for tree-based specific features


@parametrize_tree_based
def test_n_trees_of_tree_based_evaluator(
    evaluator_cls: type[FanovaImportanceEvaluator | MeanDecreaseImpurityImportanceEvaluator],
) -> None:
    study = get_study(seed=0, n_trials=3, is_multi_obj=False)
    evaluator = evaluator_cls(n_trees=10, seed=0)
    param_importance = evaluator.evaluate(study)

    evaluator = evaluator_cls(n_trees=20, seed=0)
    param_importance_different_n_trees = evaluator.evaluate(study)

    assert param_importance != param_importance_different_n_trees


@parametrize_tree_based
def test_max_depth_of_tree_based_evaluator(
    evaluator_cls: type[FanovaImportanceEvaluator | MeanDecreaseImpurityImportanceEvaluator],
) -> None:
    study = get_study(seed=0, n_trials=3, is_multi_obj=False)
    evaluator = evaluator_cls(max_depth=1, seed=0)
    param_importance = evaluator.evaluate(study)

    evaluator = evaluator_cls(max_depth=2, seed=0)
    param_importance_different_max_depth = evaluator.evaluate(study)

    assert param_importance != param_importance_different_max_depth
