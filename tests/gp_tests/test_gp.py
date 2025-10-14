from __future__ import annotations

import numpy as np
import pytest
import torch

from optuna._gp.gp import GPRegressor
from optuna._gp.gp import warn_and_convert_inf
import optuna._gp.prior as prior


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
