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
        initial_kernel_params = torch.ones(X.shape[1] + 2, dtype=torch.float64)
        gtol: float = 1e-2
        gpr = GPRegressor(
            X_train=torch.from_numpy(X),
            Y_train=torch.from_numpy(Y),
            is_categorical=torch.from_numpy(is_categorical),
            kernel_params=initial_kernel_params,
        )
        gpr._fit_kernel_params(log_prior, minimum_noise, deterministic_objective, gtol=gtol)
        assert (
            (gpr.kernel_params.inverse_squared_lengthscales != initial_kernel_params[:-2]).sum()
            + (gpr.kernel_params.kernel_scale != initial_kernel_params[-2]).sum()
            + (gpr.kernel_params.noise_var != initial_kernel_params[-1]).sum()
        )
