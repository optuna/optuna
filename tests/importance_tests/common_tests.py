from __future__ import annotations

from optuna import create_study
from optuna import Trial
from optuna.distributions import FloatDistribution
from optuna.importance import BaseImportanceEvaluator
from optuna.importance import FanovaImportanceEvaluator
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
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


def _test_n_trees_of_tree_based_evaluator(
    evaluator_cls: type[FanovaImportanceEvaluator | MeanDecreaseImpurityImportanceEvaluator],
) -> None:
    # Assumes that `seed` can be fixed to reproduce identical results.

    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = evaluator_cls(n_trees=10, seed=0)
    param_importance = evaluator.evaluate(study)

    evaluator = evaluator_cls(n_trees=20, seed=0)
    param_importance_different_n_trees = evaluator.evaluate(study)

    assert param_importance != param_importance_different_n_trees


def _test_max_depth_of_tree_based_evaluator(
    evaluator_cls: type[FanovaImportanceEvaluator | MeanDecreaseImpurityImportanceEvaluator],
) -> None:
    # Assumes that `seed` can be fixed to reproduce identical results.

    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = evaluator_cls(max_depth=1, seed=0)
    param_importance = evaluator.evaluate(study)

    evaluator = evaluator_cls(max_depth=2, seed=0)
    param_importance_different_max_depth = evaluator.evaluate(study)

    assert param_importance != param_importance_different_max_depth


def _test_evaluator_with_infinite(
    evaluator_cls: type[BaseImportanceEvaluator], inf_value: float
) -> None:
    # The test ensures that trials with infinite values are ignored to calculate importance scores.
    n_trial = 10
    seed = 13

    # Importance scores are calculated without a trial with an inf value.
    study = create_study(sampler=RandomSampler(seed=seed))
    study.optimize(objective, n_trials=n_trial)

    try:
        evaluator = evaluator_cls(seed=seed)  # type: ignore[call-arg]
    except TypeError:  # evaluator does not take seed.
        evaluator = evaluator_cls()

    param_importance_without_inf = evaluator.evaluate(study)

    # A trial with an inf value is added into the study manually.
    study.add_trial(
        create_trial(
            value=inf_value,
            params={"x1": 1.0, "x2": 1.0, "x3": 3.0},
            distributions={
                "x1": FloatDistribution(low=0.1, high=3),
                "x2": FloatDistribution(low=0.1, high=3, log=True),
                "x3": FloatDistribution(low=2, high=4, log=True),
            },
        )
    )
    # Importance scores are calculated with a trial with an inf value.
    param_importance_with_inf = evaluator.evaluate(study)

    # Obtained importance scores should be the same between with inf and without inf,
    # because the last trial whose objective value is an inf is ignored.
    # PED-ANOVA can handle inf, so anyways the length should be identical.
    assert param_importance_with_inf == param_importance_without_inf


def _test_multi_objective_evaluator_with_infinite(
    evaluator_cls: type[BaseImportanceEvaluator], target_idx: int, inf_value: float
) -> None:
    # The test ensures that trials with infinite values are ignored to calculate importance scores.
    n_trial = 10
    seed = 13

    # Importance scores are calculated without a trial with an inf value.
    study = create_study(directions=["minimize", "minimize"], sampler=RandomSampler(seed=seed))
    study.optimize(multi_objective_function, n_trials=n_trial)

    try:
        evaluator = evaluator_cls(seed=seed)  # type: ignore[call-arg]
    except TypeError:  # evaluator does not take seed.
        evaluator = evaluator_cls()

    param_importance_without_inf = evaluator.evaluate(study, target=lambda t: t.values[target_idx])

    # A trial with an inf value is added into the study manually.
    study.add_trial(
        create_trial(
            values=[inf_value, inf_value],
            params={"x1": 1.0, "x2": 1.0, "x3": 3.0},
            distributions={
                "x1": FloatDistribution(low=0.1, high=3),
                "x2": FloatDistribution(low=0.1, high=3, log=True),
                "x3": FloatDistribution(low=2, high=4, log=True),
            },
        )
    )
    # Importance scores are calculated with a trial with an inf value.
    param_importance_with_inf = evaluator.evaluate(study, target=lambda t: t.values[target_idx])

    # Obtained importance scores should be the same between with inf and without inf,
    # because the last trial whose objective value is an inf is ignored.
    # PED-ANOVA can handle inf, so anyways the length should be identical.
    assert param_importance_with_inf == param_importance_without_inf
