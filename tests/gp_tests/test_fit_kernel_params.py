from collections.abc import Generator

import numpy as np
import pytest
import torch

from optuna._gp.gp import _fit_kernel_params
from optuna._gp.gp import KernelParamsTensor
import optuna._gp.prior as prior


torch.set_printoptions(precision=10)


test_counter = 0


@pytest.fixture(autouse=True)
def track_test_counter() -> Generator[int, None, None]:
    global test_counter
    test_counter += 1
    yield test_counter


expected = [
    KernelParamsTensor(
        inverse_squared_lengthscales=torch.tensor(
            [7.5412181287, 0.4229148094], dtype=torch.float64
        ),
        kernel_scale=torch.tensor(2.4372882788, dtype=torch.float64),
        noise_var=torch.tensor(1.0000000000e-06, dtype=torch.float64),
    ),
    KernelParamsTensor(
        inverse_squared_lengthscales=torch.tensor(
            [0.9939692462, 0.9939692462, 0.6564795644], dtype=torch.float64
        ),
        kernel_scale=torch.tensor(1.4350097649, dtype=torch.float64),
        noise_var=torch.tensor(1.0000000000e-06, dtype=torch.float64),
    ),
    KernelParamsTensor(
        inverse_squared_lengthscales=torch.tensor(
            [0.6043575796, 0.6043575796], dtype=torch.float64
        ),
        kernel_scale=torch.tensor(1.4153845927, dtype=torch.float64),
        noise_var=torch.tensor(1.0000000000e-06, dtype=torch.float64),
    ),
    KernelParamsTensor(
        inverse_squared_lengthscales=torch.tensor([1.0], dtype=torch.float64),
        kernel_scale=torch.tensor(0.5080970212, dtype=torch.float64),
        noise_var=torch.tensor(1.0000000000e-06, dtype=torch.float64),
    ),
    KernelParamsTensor(
        inverse_squared_lengthscales=torch.tensor([1.0], dtype=torch.float64),
        kernel_scale=torch.tensor(0.5080970212, dtype=torch.float64),
        noise_var=torch.tensor(1.0000000000e-06, dtype=torch.float64),
    ),
    KernelParamsTensor(
        inverse_squared_lengthscales=torch.tensor(
            [7.5734182650, 0.4367414278], dtype=torch.float64
        ),
        kernel_scale=torch.tensor(2.4390349904, dtype=torch.float64),
        noise_var=torch.tensor(0.0027166261, dtype=torch.float64),
    ),
    KernelParamsTensor(
        inverse_squared_lengthscales=torch.tensor(
            [0.9930272391, 0.9930272391, 0.6515883980], dtype=torch.float64
        ),
        kernel_scale=torch.tensor(1.4324493780, dtype=torch.float64),
        noise_var=torch.tensor(0.0033516681, dtype=torch.float64),
    ),
    KernelParamsTensor(
        inverse_squared_lengthscales=torch.tensor(
            [0.6038272826, 0.6038272826], dtype=torch.float64
        ),
        kernel_scale=torch.tensor(1.4129139159, dtype=torch.float64),
        noise_var=torch.tensor(0.0033700807, dtype=torch.float64),
    ),
    KernelParamsTensor(
        inverse_squared_lengthscales=torch.tensor([1.0], dtype=torch.float64),
        kernel_scale=torch.tensor(0.5038784141, dtype=torch.float64),
        noise_var=torch.tensor(0.0032481589, dtype=torch.float64),
    ),
    KernelParamsTensor(
        inverse_squared_lengthscales=torch.tensor([1.0], dtype=torch.float64),
        kernel_scale=torch.tensor(0.5038784141, dtype=torch.float64),
        noise_var=torch.tensor(0.0032481589, dtype=torch.float64),
    ),
]


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
        initial_kernel_params = KernelParamsTensor(
            inverse_squared_lengthscales=torch.ones(X.shape[1], dtype=torch.float64),
            kernel_scale=torch.tensor(1.0, dtype=torch.float64),
            noise_var=torch.tensor(1.0, dtype=torch.float64),
        )
        gtol: float = 1e-2

        kernel_params = _fit_kernel_params(
            X=X,
            Y=Y,
            is_categorical=is_categorical,
            log_prior=log_prior,
            minimum_noise=minimum_noise,
            initial_kernel_params=initial_kernel_params,
            deterministic_objective=deterministic_objective,
            gtol=gtol,
        )
        assert torch.allclose(
            kernel_params.inverse_squared_lengthscales,
            expected[(test_counter - 1) % 10].inverse_squared_lengthscales,
        )
        assert torch.allclose(
            kernel_params.kernel_scale, expected[(test_counter - 1) % 10].kernel_scale
        )
        assert torch.allclose(kernel_params.noise_var, expected[(test_counter - 1) % 10].noise_var)
