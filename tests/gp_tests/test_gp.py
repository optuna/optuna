from __future__ import annotations

import numpy as np
import pytest
import torch

from optuna._gp.gp import ConditionalGPRegressor
from optuna._gp.gp import GPRegressor
from optuna._gp.gp import warn_and_convert_inf
import optuna._gp.prior as prior
from optuna._gp.qmc import _sample_from_normal_sobol


@pytest.mark.parametrize(
    "values,ans",
    [
        (np.array([-1, 0, 1])[:, np.newaxis], np.array([-1, 0, 1])[:, np.newaxis]),
        (
            np.array([-1, -np.inf, 0, np.inf, 1])[:, np.newaxis],
            np.array([-1, -1, 0, 1, 1])[:, np.newaxis],
        ),
        (np.array([[-1, 2], [0, -2], [1, 0]]), np.array([[-1, 2], [0, -2], [1, 0]])),
        (
            np.array([[-1, 2], [-np.inf, np.inf], [0, -np.inf], [np.inf, -2], [1, 0]]),
            np.array([[-1, 2], [-1, 2], [0, -2], [1, -2], [1, 0]]),
        ),
        (
            np.array(
                [
                    [-100, np.inf, 10],
                    [-np.inf, np.inf, 100],
                    [-10, -np.inf, np.inf],
                    [np.inf, np.inf, -np.inf],
                ]
            ),
            np.array([[-100, 0, 10], [-100, 0, 100], [-10, 0, 100], [-10, 0, 10]]),
        ),
        (np.array([-np.inf, np.inf])[:, np.newaxis], np.array([0, 0])[:, np.newaxis]),
        (np.array([])[:, np.newaxis], np.array([])[:, np.newaxis]),
    ],
)
def test_warn_and_convert_inf_for_2d_array(values: np.ndarray, ans: np.ndarray) -> None:
    assert np.allclose(warn_and_convert_inf(values), ans)


@pytest.mark.parametrize(
    "values,ans",
    [
        (np.array([-1, 0, 1]), np.array([-1, 0, 1])),
        (np.array([-1, -np.inf, 0, np.inf, 1]), np.array([-1, -1, 0, 1, 1])),
        (np.array([-np.inf, np.inf]), np.array([0, 0])),
        (np.array([]), np.array([])),
    ],
)
def test_warn_and_convert_inf_for_1d_array(values: np.ndarray, ans: np.ndarray) -> None:
    assert np.allclose(warn_and_convert_inf(values), ans)


@pytest.mark.parametrize(
    "X, Y, is_categorical",
    [
        (
            np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.1]]),
            np.array([1.0, 2.0, 3.0]),
            np.array([False, False]),
        ),
        (
            np.array([[0.1, 0.2, 0.0], [0.2, 0.3, 1.0]]),
            np.array([1.0, 2.0]),
            np.array([False, False, True]),
        ),
        (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 2.0]), np.array([True, True])),
        (np.array([[0.0]]), np.array([0.0]), np.array([True])),
        (np.array([[0.0]]), np.array([0.0]), np.array([False])),
    ],
)
@pytest.mark.parametrize("deterministic_objective", [True, False])
@pytest.mark.parametrize("torch_set_grad_enabled", [True, False])
def test_fit_kernel_params(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
    deterministic_objective: bool,
    torch_set_grad_enabled: bool,
) -> None:
    with torch.set_grad_enabled(torch_set_grad_enabled):
        log_prior = prior.default_log_prior
        minimum_noise = prior.DEFAULT_MINIMUM_NOISE_VAR
        gtol: float = 1e-2
        gpr = GPRegressor(
            X_train=torch.from_numpy(X),
            y_train=torch.from_numpy(Y),
            is_categorical=torch.from_numpy(is_categorical),
            inverse_squared_lengthscales=torch.ones(X.shape[1], dtype=torch.float64),
            kernel_scale=torch.tensor(1.0, dtype=torch.float64),
            noise_var=torch.tensor(1.0, dtype=torch.float64),
        )._fit_kernel_params(
            log_prior=log_prior,
            minimum_noise=minimum_noise,
            deterministic_objective=deterministic_objective,
            gtol=gtol,
        )

        assert (
            (gpr.inverse_squared_lengthscales != 1.0).sum()
            + (gpr.kernel_scale != 1.0).sum()
            + (gpr.noise_var != 1.0).sum()
        )


@pytest.mark.parametrize(
    "x_shape", [(1, 3), (2, 3), (1, 2, 3), (2, 1, 3), (2, 2, 3), (2, 2, 2, 3)]
)
def test_posterior(x_shape: tuple[int, ...]) -> None:
    rng = np.random.RandomState(0)
    X = rng.random(size=(10, x_shape[-1]))
    Y = rng.randn(10)
    Y = (Y - Y.mean()) / Y.std()
    log_prior = prior.default_log_prior
    minimum_noise = prior.DEFAULT_MINIMUM_NOISE_VAR
    gtol: float = 1e-2
    gpr = GPRegressor(
        X_train=torch.from_numpy(X),
        y_train=torch.from_numpy(Y),
        is_categorical=torch.from_numpy(np.zeros(X.shape[-1], dtype=bool)),
        inverse_squared_lengthscales=torch.ones(X.shape[1], dtype=torch.float64),
        kernel_scale=torch.tensor(1.0, dtype=torch.float64),
        noise_var=torch.tensor(1.0, dtype=torch.float64),
    )._fit_kernel_params(
        log_prior=log_prior,
        minimum_noise=minimum_noise,
        deterministic_objective=False,
        gtol=gtol,
    )
    x = rng.random(size=x_shape)
    mean_joint, covar = gpr.posterior(torch.from_numpy(x), joint=True)
    mean, var_ = gpr.posterior(torch.from_numpy(x), joint=False)
    assert mean_joint.shape == mean.shape and torch.allclose(mean, mean_joint)
    assert covar.shape == (*x_shape[:-1], x_shape[-2])
    assert covar.diagonal(dim1=-2, dim2=-1).shape == var_.shape and torch.allclose(
        covar.diagonal(dim1=-2, dim2=-1), var_
    ), "Diagonal Check."
    assert torch.allclose(covar, covar.transpose(-2, -1)), "Symmetric Check."
    assert torch.all(torch.det(covar) >= 0.0), "Postive Semi-definite Check."


@pytest.mark.parametrize("n_running", [1, 5])
def test_append_running_data(n_running: int) -> None:
    dim = 3
    rng = np.random.RandomState(0)
    X = torch.from_numpy(rng.random(size=(10, dim)))
    Y = torch.from_numpy(rng.randn(10))
    Y = (Y - Y.mean()) / Y.std()
    log_prior = prior.default_log_prior
    minimum_noise = prior.DEFAULT_MINIMUM_NOISE_VAR
    gtol: float = 1e-2
    gpr = GPRegressor(
        X_train=X,
        y_train=Y,
        is_categorical=torch.from_numpy(np.zeros(X.shape[-1], dtype=bool)),
        inverse_squared_lengthscales=torch.ones(X.shape[1], dtype=torch.float64),
        kernel_scale=torch.tensor(1.0, dtype=torch.float64),
        noise_var=torch.tensor(1.0, dtype=torch.float64),
    )._fit_kernel_params(
        log_prior=log_prior,
        minimum_noise=minimum_noise,
        deterministic_objective=False,
        gtol=gtol,
    )

    X_running = torch.from_numpy(rng.random(size=(n_running, dim)))
    y_running = torch.from_numpy(rng.randn(n_running))

    reference_gpr = GPRegressor(
        X_train=torch.cat([X, X_running], dim=0),
        y_train=torch.cat([Y, y_running], dim=0),
        is_categorical=torch.from_numpy(np.zeros(X.shape[-1] + n_running, dtype=bool)),
        inverse_squared_lengthscales=gpr.inverse_squared_lengthscales.clone(),
        kernel_scale=gpr.kernel_scale.clone(),
        noise_var=gpr.noise_var.clone(),
    )
    reference_gpr._cache_matrix()

    gpr.append_running_data(X_running, y_running)

    assert reference_gpr._cov_Y_Y_chol is not None
    assert gpr._cov_Y_Y_chol is not None
    assert reference_gpr._cov_Y_Y_inv_Y is not None
    assert gpr._cov_Y_Y_inv_Y is not None
    assert torch.allclose(reference_gpr._cov_Y_Y_chol, gpr._cov_Y_Y_chol)
    assert torch.allclose(reference_gpr._cov_Y_Y_inv_Y, gpr._cov_Y_Y_inv_Y)

    x = torch.from_numpy(rng.random(size=(1, dim)))
    mean, var = gpr.posterior(x)
    reference_mean, reference_var = reference_gpr.posterior(x)
    assert torch.allclose(mean, reference_mean)
    assert torch.allclose(var, reference_var)


@pytest.mark.parametrize("n_running", [1, 4])
@pytest.mark.parametrize("batch_size", [1, 16])
def test_conditional_gpr_matches_joint(n_running: int, batch_size: int) -> None:
    n_trials = 10
    dim = 3
    n_qmc_samples = 64
    stabilizing_noise = 1e-12
    X_train = torch.rand(n_trials, dim, dtype=torch.float64)
    y_train = torch.sin(X_train.sum(-1))
    gpr = GPRegressor(
        is_categorical=torch.zeros(dim, dtype=torch.bool),
        X_train=X_train,
        y_train=y_train,
        inverse_squared_lengthscales=torch.ones(dim, dtype=torch.float64),
        kernel_scale=torch.tensor(1.0, dtype=torch.float64),
        noise_var=torch.tensor(0.01, dtype=torch.float64),
    )
    gpr._cache_matrix()

    X_running = torch.rand(n_running, dim, dtype=torch.float64)
    if batch_size == 1:
        x_new = torch.rand(dim, dtype=torch.float64)
        joint_x = torch.cat([X_running, x_new.unsqueeze(0)], dim=0)
    else:
        x_new = torch.rand((batch_size, dim), dtype=torch.float64)
        joint_x = torch.cat(
            [X_running.unsqueeze(0).expand(batch_size, -1, -1), x_new.unsqueeze(1)], dim=1
        )
    qmc_seed = 42
    cond_gpr = ConditionalGPRegressor(
        gpr,
        X_running=X_running,
        n_qmc_samples=n_qmc_samples,
        qmc_seed=qmc_seed,
        stabilizing_noise=stabilizing_noise,
    )
    samples_cond = cond_gpr.sample_joint_posterior(x_new)

    mu, cov = gpr.posterior(joint_x, joint=True)
    cov.diagonal(dim1=-2, dim2=-1).add_(stabilizing_noise)
    fixed_samples = _sample_from_normal_sobol(
        dim=n_running + 1, n_samples=n_qmc_samples, seed=qmc_seed
    )
    samples_joint = mu.unsqueeze(-2) + torch.matmul(
        fixed_samples, torch.linalg.cholesky(cov).transpose(-1, -2)
    )

    torch.testing.assert_close(samples_joint, samples_cond)
