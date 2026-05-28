from __future__ import annotations

import optuna
from optuna.samplers import TPESampler


def test_union_multivariate_tpe_conditional_convergence() -> None:
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_categorical("x", [True, False])
        y = trial.suggest_float("y", -1.0, 1.0)
        if x is True:
            n = trial.suggest_categorical("n", [True, False])
            if n is True:
                a = trial.suggest_float("a", -1.0, 1.0)
                return float((a - y) ** 2 + (a + 0.75) ** 2 + 0.025)
            b = trial.suggest_float("b", -1.0, 1.0)
            return float((b - y) ** 2 + (b + 0.25) ** 2 + 0.05)
        else:
            m = trial.suggest_categorical("m", [True, False])
            if m is True:
                c = trial.suggest_float("c", -1.0, 1.0)
                return float((c - y) ** 2 + (c - 0.25) ** 2 + 0.4)
            d = trial.suggest_float("d", -1.0, 1.0)
            return float((d - y) ** 2 + (d - 0.75) ** 2 + 0.01)

    sampler = TPESampler(multivariate=True, group=False, n_startup_trials=5, seed=42)
    study = optuna.create_study(sampler=sampler, direction="minimize")

    study.enqueue_trial({"x": True, "n": True, "y": 0.0, "a": 0.0})
    study.enqueue_trial({"x": False, "m": False, "y": 0.0, "d": 0.0})

    study.optimize(objective, n_trials=15)

    assert len(study.trials) == 15
    assert all(
        t.value is not None for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )


def test_union_search_space_handling_with_nan_observations() -> None:
    sampler = TPESampler(multivariate=True, group=False, n_startup_trials=2, seed=24)
    study = optuna.create_study(sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        if trial.suggest_categorical("c", [0, 1]) == 0:
            return float(trial.suggest_float("p1", 0, 1))
        return float(trial.suggest_float("p2", 2, 3))

    study.optimize(objective, n_trials=4)
    assert len(study.trials) == 4
