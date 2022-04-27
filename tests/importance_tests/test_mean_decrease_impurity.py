import pytest

from optuna import create_study
from optuna import Trial
from optuna.distributions import FloatDistribution
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
from optuna.samplers import RandomSampler
from optuna.trial import create_trial


def objective(trial: Trial) -> float:
    x1 = trial.suggest_float("x1", 0.1, 3)
    x2 = trial.suggest_float("x2", 0.1, 3, log=True)
    x3 = trial.suggest_float("x3", 2, 4, log=True)
    return x1 + x2 * x3


def test_mean_decrease_impurity_importance_evaluator_n_trees() -> None:
    # Assumes that `seed` can be fixed to reproduce identical results.

    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = MeanDecreaseImpurityImportanceEvaluator(n_trees=10, seed=0)
    param_importance = evaluator.evaluate(study)

    evaluator = MeanDecreaseImpurityImportanceEvaluator(n_trees=20, seed=0)
    param_importance_different_n_trees = evaluator.evaluate(study)

    assert param_importance != param_importance_different_n_trees


def test_mean_decrease_impurity_importance_evaluator_max_depth() -> None:
    # Assumes that `seed` can be fixed to reproduce identical results.

    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = MeanDecreaseImpurityImportanceEvaluator(max_depth=1, seed=0)
    param_importance = evaluator.evaluate(study)

    evaluator = MeanDecreaseImpurityImportanceEvaluator(max_depth=2, seed=0)
    param_importance_different_max_depth = evaluator.evaluate(study)

    assert param_importance != param_importance_different_max_depth


def test_mean_decrease_impurity_importance_evaluator_seed() -> None:
    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = MeanDecreaseImpurityImportanceEvaluator(seed=2)
    param_importance = evaluator.evaluate(study)

    evaluator = MeanDecreaseImpurityImportanceEvaluator(seed=2)
    param_importance_same_seed = evaluator.evaluate(study)
    assert param_importance == param_importance_same_seed

    evaluator = MeanDecreaseImpurityImportanceEvaluator(seed=3)
    param_importance_different_seed = evaluator.evaluate(study)
    assert param_importance != param_importance_different_seed


def test_mean_decrease_impurity_importance_evaluator_with_target() -> None:
    # Assumes that `seed` can be fixed to reproduce identical results.

    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = MeanDecreaseImpurityImportanceEvaluator(seed=0)
    param_importance = evaluator.evaluate(study)
    param_importance_with_target = evaluator.evaluate(
        study,
        target=lambda t: t.params["x1"] + t.params["x2"],
    )

    assert param_importance != param_importance_with_target


@pytest.mark.parametrize("inf_value", [float("inf"), -float("inf")])
def test_mean_decrease_impurity_importance_evaluator_with_infinite(inf_value: float) -> None:
    # The test ensures that trials with infinite values are ignored to calculate importance scores.
    n_trial = 10
    trial_sampling_seed = 13
    evaluator_seed = 67

    # Importance scores are calculated without a trial with an inf value.
    study_without_inf = create_study(sampler=RandomSampler(seed=trial_sampling_seed))
    study_without_inf.optimize(objective, n_trials=n_trial)

    evaluator_without_inf = MeanDecreaseImpurityImportanceEvaluator(seed=evaluator_seed)
    param_importance_without_inf = evaluator_without_inf.evaluate(study_without_inf)

    # Importance scores are calculated with a trial with an inf value.
    study_with_inf = create_study(sampler=RandomSampler(seed=trial_sampling_seed))
    study_with_inf.optimize(objective, n_trials=n_trial)
    # A trial with an inf value is added into the study manually.
    study_with_inf.add_trial(
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
    evaluator_with_inf = MeanDecreaseImpurityImportanceEvaluator(seed=evaluator_seed)
    param_importance_with_inf = evaluator_with_inf.evaluate(study_with_inf)

    # Obtained importance scores should be same among with inf and without inf,
    # if an trial with an inf value is ignored.
    assert param_importance_with_inf == param_importance_without_inf
