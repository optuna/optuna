from typing import Tuple

import pytest

from optuna import create_study
from optuna import Trial
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.integration.shap import ShapleyImportanceEvaluator
from optuna.samplers import RandomSampler
from optuna.trial import create_trial


pytestmark = pytest.mark.integration


def objective(trial: Trial) -> float:
    x1: float = trial.suggest_float("x1", 0.1, 3)
    x2: float = trial.suggest_float("x2", 0.1, 3, log=True)
    x3: int = trial.suggest_int("x3", 2, 4, log=True)
    x4: float = trial.suggest_categorical("x4", [0.1, 1.0, 10.0])
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
        x4: float = trial.suggest_categorical("x4", [0.1, 1.0, 10.0])
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
