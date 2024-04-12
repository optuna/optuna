from __future__ import annotations

from _pytest.logging import LogCaptureFixture

import optuna


def test_after_convergence(caplog: LogCaptureFixture) -> None:
    caplog.clear()
    optuna.logging.enable_propagation()
    sampler = optuna.samplers.GPSampler(seed=0)
    study = optuna.create_study(sampler=sampler)
    dists: dict[str, optuna.distributions.BaseDistribution] = {
        "x": optuna.distributions.FloatDistribution(0.0, 1.0)
    }
    uniform_trials = [
        optuna.create_trial(value=(i + 1) / 10, params={"x": (i + 1) / 10}, distributions=dists)
        for i in range(10)
    ]
    uniform_near_optimal_trials = [
        optuna.create_trial(value=(i + 1) / 1e5, params={"x": (i + 1) / 1e5}, distributions=dists)
        for i in range(20)
    ]
    optimal_trials = [
        optuna.create_trial(value=0.0, params={"x": 0.0}, distributions=dists) for _ in range(10)
    ]
    # A large `optimal_trials` causes the instability in the kernel inversion, leading to
    # instability in the variance calculation.
    study.add_trials(uniform_trials + uniform_near_optimal_trials + optimal_trials)
    study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=1)
    assert len(caplog.text) > 0, "To the PR author, did you change the kernel implementation?"
