from __future__ import annotations

from collections.abc import Sequence
import importlib
import sys
import warnings

from _pytest.logging import LogCaptureFixture
import numpy as np
import pytest

import optuna
import optuna._gp.acqf as acqf_module
import optuna._gp.gp as optuna_gp
import optuna._gp.optim_mixed as optim_mixed
import optuna._gp.prior as prior
import optuna._gp.search_space as gp_search_space
from optuna.samplers import GPSampler
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.trial import FrozenTrial


def test_after_convergence(caplog: LogCaptureFixture) -> None:
    # A large `optimal_trials` causes the instability in the kernel inversion, leading to
    # instability in the variance calculation.
    X_uniform = [(i + 1) / 10 for i in range(10)]
    X_uniform_near_optimal = [(i + 1) / 1e5 for i in range(20)]
    X_optimal = [0.0] * 10
    X = np.array(X_uniform + X_uniform_near_optimal + X_optimal)
    score_vals = -(X - np.mean(X)) / np.std(X)
    search_space = gp_search_space.SearchSpace(
        {"a": optuna.distributions.FloatDistribution(0.0, 1.0)}
    )
    gpr = optuna_gp.fit_kernel_params(
        X=X[:, np.newaxis],
        Y=score_vals,
        is_categorical=np.array([False]),
        log_prior=prior.default_log_prior,
        minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=False,
    )
    acqf_params = acqf_module.LogEI(
        gpr=gpr, search_space=search_space, threshold=np.max(score_vals)
    )
    caplog.clear()
    optuna.logging.enable_propagation()
    optim_mixed.optimize_acqf_mixed(acqf_params, rng=np.random.RandomState(42))
    # len(caplog.text) > 0 means the optimization has already converged.
    assert len(caplog.text) > 0, "Did you change the kernel implementation?"


@pytest.mark.parametrize("constraint_value", [-1.0, 0.0, 1.0, -float("inf"), float("inf")])
@pytest.mark.parametrize("n_objectives", [1, 2])
@pytest.mark.filterwarnings("ignore:.*GPSampler cannot handle infinite values*")
def test_constraints_func(constraint_value: float, n_objectives: int) -> None:
    n_trials = 5
    constraints_func_call_count = 0

    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        nonlocal constraints_func_call_count
        constraints_func_call_count += 1

        return (constraint_value + trial.number,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = GPSampler(n_startup_trials=2, constraints_func=constraints_func)

    def objective(trial: optuna.Trial) -> float | tuple[float, float]:
        x = trial.suggest_float("x", 0, 1)
        if n_objectives == 1:
            return x
        else:
            return x, (x - 2) ** 2

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    assert len(study.trials) == n_trials
    assert constraints_func_call_count == n_trials
    for trial in study.trials:
        for x, y in zip(trial.system_attrs[_CONSTRAINTS_KEY], (constraint_value + trial.number,)):
            assert x == y


@pytest.mark.parametrize("n_objectives", [1, 2])
def test_constraints_func_nan(n_objectives: int) -> None:
    n_trials = 5

    def constraints_func(_: FrozenTrial) -> Sequence[float]:
        return (float("nan"),)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = GPSampler(n_startup_trials=2, constraints_func=constraints_func)

    def objective(
        trial: optuna.Trial | optuna.trial.FrozenTrial,
    ) -> tuple[float] | tuple[float, float]:
        x = trial.suggest_float("x", 0, 1)
        if n_objectives == 1:
            return (x,)
        else:
            return x, (x - 2) ** 2

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler)
    with pytest.raises(ValueError):
        study.optimize(objective, n_trials=n_trials)

    trials = study.get_trials()
    assert len(trials) == 1  # The error stops optimization, but completed trials are recorded.
    assert all(0 <= x <= 1 for x in trials[0].params.values())  # The params are normal.
    assert trials[0].values == list(objective(trials[0]))  # The values are normal.
    assert trials[0].system_attrs[_CONSTRAINTS_KEY] is None  # None is set for constraints.


def test_behavior_without_greenlet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "greenlet", None)
    import optuna._gp.batched_lbfgsb as optimization_module

    importlib.reload(optimization_module)
    assert optimization_module._greenlet_imports.is_successful() is False

    # See if optimization still works without greenlet
    import optuna

    sampler = optuna.samplers.GPSampler(seed=42)
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda trial: trial.suggest_float("x", -10, 10) ** 2, n_trials=15)
