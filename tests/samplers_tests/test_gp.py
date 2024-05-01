from __future__ import annotations

from _pytest.logging import LogCaptureFixture
import numpy as np

import optuna
import optuna._gp.acqf as acqf
import optuna._gp.optim_mixed as optim_mixed
import optuna._gp.prior as prior
import optuna._gp.search_space as gp_search_space


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
