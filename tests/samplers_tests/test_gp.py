from __future__ import annotations

from collections.abc import Sequence
import warnings

from _pytest.logging import LogCaptureFixture
import numpy as np
import pytest

import optuna
import optuna._gp.acqf as acqf
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
        scale_types=np.array([gp_search_space.ScaleType.LINEAR]),
        bounds=np.array([[0.0, 1.0]]),
        steps=np.zeros(1, dtype=float),
    )
    kernel_params = optuna._gp.gp.fit_kernel_params(
        X=X[:, np.newaxis],
        Y=score_vals,
        is_categorical=np.array([False]),
        log_prior=prior.default_log_prior,
        minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=False,
    )
    acqf_params = acqf.create_acqf_params(
        acqf_type=acqf.AcquisitionFunctionType.LOG_EI,
        kernel_params=kernel_params,
        search_space=search_space,
        X=X[:, np.newaxis],
        Y=score_vals,
    )
    caplog.clear()
    optuna.logging.enable_propagation()
    optim_mixed.optimize_acqf_mixed(acqf_params, rng=np.random.RandomState(42))
    # len(caplog.text) > 0 means the optimization has already converged.
    assert len(caplog.text) > 0, "Did you change the kernel implementation?"


@pytest.mark.parametrize("constraint_value", [-1.0, 0.0, 1.0, -float("inf"), float("inf")])
@pytest.mark.filterwarnings("ignore:.*GPSampler cannot handle infinite values*")
def test_constraints_func(constraint_value: float) -> None:
    n_trials = 5
    constraints_func_call_count = 0

    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        nonlocal constraints_func_call_count
        constraints_func_call_count += 1

        return (constraint_value + trial.number,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = GPSampler(n_startup_trials=2, constraints_func=constraints_func)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=n_trials)

    assert len(study.trials) == n_trials
    assert constraints_func_call_count == n_trials
    for trial in study.trials:
        for x, y in zip(trial.system_attrs[_CONSTRAINTS_KEY], (constraint_value + trial.number,)):
            assert x == y


def test_constraints_func_nan() -> None:
    n_trials = 5

    def constraints_func(_: FrozenTrial) -> Sequence[float]:
        return (float("nan"),)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = GPSampler(n_startup_trials=2, constraints_func=constraints_func)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    with pytest.raises(ValueError):
        study.optimize(
            lambda t: t.suggest_float("x", 0, 1),
            n_trials=n_trials,
        )

    trials = study.get_trials()
    assert len(trials) == 1  # The error stops optimization, but completed trials are recorded.
    assert all(0 <= x <= 1 for x in trials[0].params.values())  # The params are normal.
    assert trials[0].values == list(trials[0].params.values())  # The values are normal.
    assert trials[0].system_attrs[_CONSTRAINTS_KEY] is None  # None is set for constraints.
