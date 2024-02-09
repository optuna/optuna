from __future__ import annotations

import pytest

from optuna import create_study
from optuna import Study
from optuna import Trial
from optuna.distributions import FloatDistribution
from optuna.importance import BaseImportanceEvaluator
from optuna.importance import FanovaImportanceEvaluator
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
from optuna.importance import PedAnovaImportanceEvaluator
from optuna.samplers import RandomSampler
from optuna.trial import create_trial


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


@pytest.mark.parametrize(
    "evaluator_cls", (FanovaImportanceEvaluator, MeanDecreaseImpurityImportanceEvaluator)
)
def _test_n_trees_of_tree_based_evaluator(
    evaluator_cls: type[FanovaImportanceEvaluator | MeanDecreaseImpurityImportanceEvaluator],
) -> None:
    study = get_study(seed=0, n_trials=3, is_multi_obj=False)
    evaluator = evaluator_cls(n_trees=10, seed=0)
    param_importance = evaluator.evaluate(study)

    evaluator = evaluator_cls(n_trees=20, seed=0)
    param_importance_different_n_trees = evaluator.evaluate(study)

    assert param_importance != param_importance_different_n_trees


@pytest.mark.parametrize(
    "evaluator_cls", (FanovaImportanceEvaluator, MeanDecreaseImpurityImportanceEvaluator)
)
def _test_max_depth_of_tree_based_evaluator(
    evaluator_cls: type[FanovaImportanceEvaluator | MeanDecreaseImpurityImportanceEvaluator],
) -> None:
    study = get_study(seed=0, n_trials=3, is_multi_obj=False)
    evaluator = evaluator_cls(max_depth=1, seed=0)
    param_importance = evaluator.evaluate(study)

    evaluator = evaluator_cls(max_depth=2, seed=0)
    param_importance_different_max_depth = evaluator.evaluate(study)

    assert param_importance != param_importance_different_max_depth


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


def _test_evaluator_with_infinite(
    evaluator_cls: type[BaseImportanceEvaluator], inf_value: float, target_idx: int | None = None
) -> None:
    # The test ensures that trials with infinite values are ignored to calculate importance scores.
    is_multi_obj = target_idx is not None
    study = get_study(seed=13, n_trials=10, is_multi_obj=is_multi_obj)

    try:
        evaluator = evaluator_cls(seed=13)  # type: ignore[call-arg]
    except TypeError:  # evaluator does not take seed.
        evaluator = evaluator_cls()

    if is_multi_obj:
        assert target_idx is not None
        target = lambda t: t.values[target_idx]  # noqa: E731
    else:
        target = None

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


def test_error_in_ped_anova() -> None:
    with pytest.raises(RuntimeError, match=r".*multi-objective optimization.*"):
        evaluator = PedAnovaImportanceEvaluator()
        study = get_study(seed=0, n_trials=5, is_multi_obj=True)
        evaluator.evaluate(study)


@pytest.mark.parametrize(
    "evaluator_cls",
    (
        FanovaImportanceEvaluator,
        MeanDecreaseImpurityImportanceEvaluator,
        PedAnovaImportanceEvaluator,
    ),
)
@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
def test_importance_evaluator_with_infinite(
    evaluator_cls: type[BaseImportanceEvaluator], inf_value: float
) -> None:
    _test_evaluator_with_infinite(evaluator_cls, inf_value)


@pytest.mark.parametrize(
    "evaluator_cls",
    (
        FanovaImportanceEvaluator,
        MeanDecreaseImpurityImportanceEvaluator,
        PedAnovaImportanceEvaluator,
    ),
)
@pytest.mark.parametrize("target_idx", [0, 1])
@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
def test_multi_objective_importance_evaluator_with_infinite(
    evaluator_cls: type[BaseImportanceEvaluator], target_idx: int, inf_value: float
) -> None:
    _test_evaluator_with_infinite(evaluator_cls, inf_value, target_idx)