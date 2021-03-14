from optuna import create_study
from optuna import Trial
from optuna.importance import FanovaImportanceEvaluator
from optuna.samplers import RandomSampler


def objective(trial: Trial) -> float:
    x1 = trial.suggest_float("x1", 0.1, 3)
    x2 = trial.suggest_float("x2", 0.1, 3, log=True)
    x3 = trial.suggest_float("x3", 2, 4, log=True)
    return x1 + x2 * x3


def test_fanova_importance_evaluator_n_trees() -> None:
    # Assumes that `seed` can be fixed to reproduce identical results.

    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = FanovaImportanceEvaluator(n_trees=10, seed=0)
    param_importance = evaluator.evaluate(study)

    evaluator = FanovaImportanceEvaluator(n_trees=20, seed=0)
    param_importance_different_n_trees = evaluator.evaluate(study)

    assert param_importance != param_importance_different_n_trees


def test_fanova_importance_evaluator_max_depth() -> None:
    # Assumes that `seed` can be fixed to reproduce identical results.

    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = FanovaImportanceEvaluator(max_depth=1, seed=0)
    param_importance = evaluator.evaluate(study)

    evaluator = FanovaImportanceEvaluator(max_depth=2, seed=0)
    param_importance_different_max_depth = evaluator.evaluate(study)

    assert param_importance != param_importance_different_max_depth


def test_fanova_importance_evaluator_seed() -> None:
    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = FanovaImportanceEvaluator(seed=2)
    param_importance = evaluator.evaluate(study)

    evaluator = FanovaImportanceEvaluator(seed=2)
    param_importance_same_seed = evaluator.evaluate(study)
    assert param_importance == param_importance_same_seed

    evaluator = FanovaImportanceEvaluator(seed=3)
    param_importance_different_seed = evaluator.evaluate(study)
    assert param_importance != param_importance_different_seed


def test_fanova_importance_evaluator_with_target() -> None:
    # Assumes that `seed` can be fixed to reproduce identical results.

    study = create_study(sampler=RandomSampler(seed=0))
    study.optimize(objective, n_trials=3)

    evaluator = FanovaImportanceEvaluator(seed=0)
    param_importance = evaluator.evaluate(study)
    param_importance_with_target = evaluator.evaluate(
        study,
        target=lambda t: t.params["x1"] + t.params["x2"],
    )

    assert param_importance != param_importance_with_target
