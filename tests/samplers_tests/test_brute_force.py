import pytest

import optuna


def test_brute_force_sampler_fixed_search_space() -> None:
    # Test that the BruteForceSampler samples all combinations in a fixed search space.
    # This test validates the core functionality of the new implementation.

    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_categorical("x", ["a", "b"])
        y = trial.suggest_int("y", 0, 1)
        z = trial.suggest_float("z", 0.1, 0.2, step=0.1)
        return 1.0 if (x == "b" and y == 1 and z > 0.15) else 0.0

    study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler(seed=0))
    study.optimize(objective, n_trials=8)  # 2 * 2 * 2 = 8 combinations

    # All 8 possible combinations should have been sampled.
    assert len(study.trials) == 8

    # Check that the best trial is correct.
    assert study.best_value == 1.0
    assert study.best_params["x"] == "b"
    assert study.best_params["y"] == 1
    assert study.best_params["z"] > 0.15


def test_brute_force_sampler_reshuffle() -> None:
    # Test that the sampler reshuffles and continues sampling after exhausting all combinations.

    def objective(trial: optuna.trial.Trial) -> float:
        trial.suggest_categorical("x", ["a", "b"])
        return 0.0

    # There are 2 combinations. We run 5 trials to test the reshuffling.
    study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler(seed=0))
    study.optimize(objective, n_trials=5)

    assert len(study.trials) == 5


def test_unsupported_distribution() -> None:
    # Test that an error is raised for unsupported distribution types during grid creation.

    def objective_float(trial: optuna.trial.Trial) -> float:
        # The sampler's first trial runs fine, but `after_trial` will fail.
        trial.suggest_float("x", 0.0, 1.0)  # No step provided, should fail.
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler())
    with pytest.raises(ValueError, match="FloatDistribution must have a step"):
        study.optimize(objective_float, n_trials=1)
