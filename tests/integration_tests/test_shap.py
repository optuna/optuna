from collections import OrderedDict
from typing import List
from typing import Tuple

import pytest

import optuna
from optuna import create_study
from optuna import samplers
from optuna import Trial
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.integration.shap import ShapleyImportanceEvaluator
from optuna.samplers import RandomSampler
from optuna.testing.objectives import pruned_objective
from optuna.testing.storages import STORAGE_MODES
from optuna.testing.storages import StorageSupplier
from optuna.trial import create_trial


def objective(trial: Trial) -> float:
    x1: float = trial.suggest_float("x1", 0.1, 3)
    x2: float = trial.suggest_float("x2", 0.1, 3, log=True)
    x3: int = trial.suggest_int("x3", 2, 4, log=True)
    x4 = trial.suggest_categorical("x4", [0.1, 1.0, 10.0])
    assert isinstance(x4, float)
    return x1 + x2 * x3 + x4


def test_mean_abs_shap_importance_evaluator_n_trees() -> None:
    # Assumes that `seed` can be fixed to reproduce identical results.

    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = ShapleyImportanceEvaluator(n_trees=10, seed=0)
    param_importance = evaluator.evaluate(study)

    evaluator = ShapleyImportanceEvaluator(n_trees=20, seed=0)
    param_importance_different_n_trees = evaluator.evaluate(study)

    assert param_importance != param_importance_different_n_trees


def test_mean_abs_shap_importance_evaluator_max_depth() -> None:
    # Assumes that `seed` can be fixed to reproduce identical results.

    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = ShapleyImportanceEvaluator(max_depth=1, seed=0)
    param_importance = evaluator.evaluate(study)

    evaluator = ShapleyImportanceEvaluator(max_depth=2, seed=0)
    param_importance_different_max_depth = evaluator.evaluate(study)

    assert param_importance != param_importance_different_max_depth


def test_mean_abs_shap_importance_evaluator_seed() -> None:
    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = ShapleyImportanceEvaluator(seed=2)
    param_importance = evaluator.evaluate(study)

    evaluator = ShapleyImportanceEvaluator(seed=2)
    param_importance_same_seed = evaluator.evaluate(study)
    assert param_importance == param_importance_same_seed

    evaluator = ShapleyImportanceEvaluator(seed=3)
    param_importance_different_seed = evaluator.evaluate(study)
    assert param_importance != param_importance_different_seed


def test_mean_abs_shap_importance_evaluator_with_target() -> None:
    # Assumes that `seed` can be fixed to reproduce identical results.

    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = ShapleyImportanceEvaluator(seed=0)
    param_importance = evaluator.evaluate(study)
    param_importance_with_target = evaluator.evaluate(
        study,
        target=lambda t: t.params["x1"] + t.params["x2"],
    )

    assert param_importance != param_importance_with_target


@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
def test_shap_importance_evaluator_with_infinite(inf_value: float) -> None:
    # The test ensures that trials with infinite values are ignored to calculate importance scores.
    n_trial = 10
    seed = 13

    # Importance scores are calculated without a trial with an inf value.
    study = create_study(sampler=RandomSampler(seed=seed))
    study.optimize(objective, n_trials=n_trial)

    evaluator = ShapleyImportanceEvaluator(seed=seed)
    param_importance_without_inf = evaluator.evaluate(study)

    # A trial with an inf value is added into the study manually.
    study.add_trial(
        create_trial(
            value=inf_value,
            params={"x1": 1.0, "x2": 1.0, "x3": 3.0, "x4": 0.1},
            distributions={
                "x1": FloatDistribution(low=0.1, high=3),
                "x2": FloatDistribution(low=0.1, high=3, log=True),
                "x3": IntDistribution(low=2, high=4, log=True),
                "x4": CategoricalDistribution([0.1, 1, 10]),
            },
        )
    )
    # Importance scores are calculated with a trial with an inf value.
    param_importance_with_inf = evaluator.evaluate(study)

    # Obtained importance scores should be the same between with inf and without inf,
    # because the last trial whose objective value is an inf is ignored.
    assert param_importance_with_inf == param_importance_without_inf


@pytest.mark.parametrize("target_idx", [0, 1])
@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
def test_multi_objective_shap_importance_evaluator_with_infinite(
    target_idx: int, inf_value: float
) -> None:
    def multi_objective_function(trial: Trial) -> Tuple[float, float]:
        x1: float = trial.suggest_float("x1", 0.1, 3)
        x2: float = trial.suggest_float("x2", 0.1, 3, log=True)
        x3: int = trial.suggest_int("x3", 2, 4, log=True)
        x4 = trial.suggest_categorical("x4", [0.1, 1.0, 10.0])
        assert isinstance(x4, float)
        return (x1 + x2 * x3 + x4, x1 * x4)

    # The test ensures that trials with infinite values are ignored to calculate importance scores.
    n_trial = 10
    seed = 13

    # Importance scores are calculated without a trial with an inf value.
    study = create_study(directions=["minimize", "minimize"], sampler=RandomSampler(seed=seed))
    study.optimize(multi_objective_function, n_trials=n_trial)

    evaluator = ShapleyImportanceEvaluator(seed=seed)
    param_importance_without_inf = evaluator.evaluate(study, target=lambda t: t.values[target_idx])

    # A trial with an inf value is added into the study manually.
    study.add_trial(
        create_trial(
            values=[inf_value, inf_value],
            params={"x1": 1.0, "x2": 1.0, "x3": 3.0, "x4": 0.1},
            distributions={
                "x1": FloatDistribution(low=0.1, high=3),
                "x2": FloatDistribution(low=0.1, high=3, log=True),
                "x3": IntDistribution(low=2, high=4, log=True),
                "x4": CategoricalDistribution([0.1, 1, 10]),
            },
        )
    )
    # Importance scores are calculated with a trial with an inf value.
    param_importance_with_inf = evaluator.evaluate(study, target=lambda t: t.values[target_idx])

    # Obtained importance scores should be the same between with inf and without inf,
    # because the last trial whose objective value is an inf is ignored.
    assert param_importance_with_inf == param_importance_without_inf


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_param_importance_target_is_none_and_study_is_multi_obj(
    storage_mode: str,
) -> None:
    def objective(trial: Trial) -> Tuple[float, float]:
        x1 = trial.suggest_float("x1", 0.1, 3)
        x2 = trial.suggest_float("x2", 0.1, 3, log=True)
        x3 = trial.suggest_float("x3", 0, 3, step=1)
        x4 = trial.suggest_int("x4", -3, 3)
        x5 = trial.suggest_int("x5", 1, 5, log=True)
        x6 = trial.suggest_categorical("x6", [1.0, 1.1, 1.2])
        if trial.number % 2 == 0:
            # Conditional parameters are ignored unless `params` is specified and is not `None`.
            x7 = trial.suggest_float("x7", 0.1, 3)

        assert isinstance(x6, float)
        value = x1**4 + x2 + x3 - x4**2 - x5 + x6
        if trial.number % 2 == 0:
            value += x7
        return value, 0.0

    with StorageSupplier(storage_mode) as storage:
        study = create_study(directions=["minimize", "minimize"], storage=storage)
        study.optimize(objective, n_trials=3)

        with pytest.raises(ValueError):
            ShapleyImportanceEvaluator().evaluate(study)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_params_importance(
    storage_mode: str,
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

        assert isinstance(x6, float)
        value = x1**4 + x2 + x3 - x4**2 - x5 + x6
        if trial.number % 2 == 0:
            value += x7
        return value

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage, sampler=samplers.RandomSampler())
        study.optimize(objective, n_trials=3)

        param_importance = ShapleyImportanceEvaluator().evaluate(study)

        assert isinstance(param_importance, OrderedDict)
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


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize("params", [[], ["x1"], ["x1", "x3"], ["x1", "x4"]])
def test_get_param_importances_with_params(
    storage_mode: str,
    params: List[str],
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

        param_importance = ShapleyImportanceEvaluator().evaluate(study, params=params)

        assert isinstance(param_importance, OrderedDict)
        assert len(param_importance) == len(params)
        assert all(param in param_importance for param in params)
        for param_name, importance in param_importance.items():
            assert isinstance(param_name, str)
            assert isinstance(importance, float)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_param_importances_with_target(
    storage_mode: str,
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

        param_importance = ShapleyImportanceEvaluator().evaluate(
            study,
            target=lambda t: t.params["x1"] + t.params["x2"],
        )

        assert isinstance(param_importance, OrderedDict)
        assert len(param_importance) == 3
        assert all(param_name in param_importance for param_name in ["x1", "x2", "x3"])
        prev_importance = float("inf")
        for param_name, importance in param_importance.items():
            assert isinstance(param_name, str)
            assert isinstance(importance, float)
            assert importance <= prev_importance
            prev_importance = importance


def test_get_param_importances_invalid_empty_study() -> None:

    study = create_study()

    with pytest.raises(ValueError):
        ShapleyImportanceEvaluator().evaluate(study)

    study.optimize(pruned_objective, n_trials=3)

    with pytest.raises(ValueError):
        ShapleyImportanceEvaluator().evaluate(study)


def test_get_param_importances_invalid_single_trial() -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, 3)
        return x1**2

    study = create_study()
    study.optimize(objective, n_trials=1)

    with pytest.raises(ValueError):
        ShapleyImportanceEvaluator().evaluate(study)


def test_get_param_importances_invalid_no_completed_trials_params() -> None:
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
        ShapleyImportanceEvaluator().evaluate(study, params=["x2"])

    # None of the trials with `x2` are completed. Adding "x1" should not matter.
    with pytest.raises(ValueError):
        ShapleyImportanceEvaluator().evaluate(study, params=["x1", "x2"])

    # None of the trials contain `x3`.
    with pytest.raises(ValueError):
        ShapleyImportanceEvaluator().evaluate(study, params=["x3"])


def test_get_param_importances_invalid_dynamic_search_space_params() -> None:
    def objective(trial: Trial) -> float:
        x1 = trial.suggest_float("x1", 0.1, trial.number + 0.1)
        return x1**2

    study = create_study()
    study.optimize(objective, n_trials=3)

    with pytest.raises(ValueError):
        ShapleyImportanceEvaluator().evaluate(study, params=["x1"])


def test_get_param_importances_empty_search_space() -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_float("x", 0, 5)
        y = trial.suggest_float("y", 1, 1)
        return 4 * x**2 + 4 * y**2

    study = create_study()
    study.optimize(objective, n_trials=3)

    param_importance = ShapleyImportanceEvaluator().evaluate(study)

    assert len(param_importance) == 2
    assert all([param in param_importance for param in ["x", "y"]])
    assert param_importance["y"] == 0.0
