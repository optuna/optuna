from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from optuna._gp.gp import _get_raw_param_bounds
from optuna._gp.gp import ConditionalGPRegressor
from optuna._gp.gp import fit_kernel_params
from optuna._gp.gp import GPRegressor
from optuna._gp.gp import Matern52Kernel
from optuna._gp.gp import RBFKernel
from optuna._gp.gp import warn_and_convert_inf
import optuna._gp.prior as prior
from optuna._gp.qmc import sample_from_normal_sobol


if TYPE_CHECKING:
    from collections.abc import Callable

    from optuna._gp.gp import KernelChoiceType


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
@pytest.mark.parametrize("kernel_type", ["matern", "rbf"])
def test_fit_kernel_params(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
    deterministic_objective: bool,
    torch_set_grad_enabled: bool,
    kernel_type: KernelChoiceType,
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
            kernel_type=kernel_type,
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
@pytest.mark.parametrize("kernel_type", ["matern", "rbf"])
def test_posterior(x_shape: tuple[int, ...], kernel_type: KernelChoiceType) -> None:
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
        kernel_type=kernel_type,
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
@pytest.mark.parametrize("kernel_type", ["matern", "rbf"])
def test_append_running_data(n_running: int, kernel_type: KernelChoiceType) -> None:
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
        kernel_type=kernel_type,
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
        kernel_type=kernel_type,
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
@pytest.mark.parametrize("kernel_type", ["matern", "rbf"])
def test_conditional_gpr_matches_joint(
    n_running: int, batch_size: int, kernel_type: KernelChoiceType
) -> None:
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
        kernel_type=kernel_type,
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
    fixed_samples = sample_from_normal_sobol(
        dim=n_running + 1, n_samples=n_qmc_samples, seed=qmc_seed
    )
    samples_joint = mu.unsqueeze(-2) + torch.matmul(
        fixed_samples, torch.linalg.cholesky(cov).transpose(-1, -2)
    )

    torch.testing.assert_close(samples_joint, samples_cond)


def _matern52_reference(squared_distance: torch.Tensor) -> torch.Tensor:
    sqrt5d = torch.sqrt(5 * squared_distance)
    return torch.exp(-sqrt5d) * (5 / 3 * squared_distance + sqrt5d + 1)


# The derivatives w.r.t. squared_distance at squared_distance = 0, where the references above are
# not differentiable by PyTorch, so the kernels save the derivatives manually.
KERNEL_TEST_CASES = [
    (RBFKernel, lambda d2: torch.exp(-0.5 * d2), -0.5),
    (Matern52Kernel, _matern52_reference, -5 / 6),
]


@pytest.mark.parametrize("kernel_cls, reference, deriv_at_zero", KERNEL_TEST_CASES)
def test_kernel_value_and_derivative_w_r_t_squared_distance(
    kernel_cls: type[torch.autograd.Function],
    reference: Callable[[torch.Tensor], torch.Tensor],
    deriv_at_zero: float,
) -> None:
    squared_distance = torch.tensor([1e-8, 0.1, 1.0, 10.0], dtype=torch.float64)
    x = squared_distance.clone().requires_grad_(True)
    value = kernel_cls.apply(x)  # type: ignore[no-untyped-call]
    value.sum().backward()

    x_ref = squared_distance.clone().requires_grad_(True)
    value_ref = reference(x_ref)
    value_ref.sum().backward()  # type: ignore[no-untyped-call]

    assert x.grad is not None and x_ref.grad is not None
    torch.testing.assert_close(value, value_ref)
    torch.testing.assert_close(x.grad, x_ref.grad)


@pytest.mark.parametrize("kernel_cls, _, deriv_at_zero", KERNEL_TEST_CASES)
def test_kernel_value_and_derivative_at_zero_squared_distance(
    kernel_cls: type[torch.autograd.Function],
    _: Callable[[torch.Tensor], torch.Tensor],
    deriv_at_zero: float,
) -> None:
    x = torch.zeros(1, dtype=torch.float64).requires_grad_(True)
    value = kernel_cls.apply(x)  # type: ignore[no-untyped-call]
    value.sum().backward()

    assert x.grad is not None
    torch.testing.assert_close(value, torch.ones(1, dtype=torch.float64))
    torch.testing.assert_close(x.grad, torch.full((1,), deriv_at_zero, dtype=torch.float64))


@pytest.mark.parametrize("kernel_type", ["matern", "rbf"])
def test_kernel_matrix_is_positive_definite(kernel_type: KernelChoiceType) -> None:
    rng = np.random.RandomState(0)
    X = torch.from_numpy(rng.random(size=(8, 3)))
    gpr = GPRegressor(
        is_categorical=torch.zeros(3, dtype=torch.bool),
        X_train=X,
        y_train=torch.from_numpy(rng.randn(8)),
        inverse_squared_lengthscales=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        kernel_scale=torch.tensor(2.0, dtype=torch.float64),
        noise_var=torch.tensor(1e-6, dtype=torch.float64),
        kernel_type=kernel_type,
    )
    cov = gpr.kernel()
    # kernel(x, x) = kernel_scale must hold since the kernels return 1.0 at zero distance.
    torch.testing.assert_close(cov.diagonal(), torch.full((8,), 2.0, dtype=torch.float64))
    assert torch.all(torch.linalg.eigvalsh(cov) > 0.0)


@pytest.mark.parametrize(
    "bounds, ans_lower, ans_upper",
    [
        (prior.DEFAULT_KERNEL_PARAM_BOUNDS, [-math.inf] * 4, [math.inf] * 4),
        # lengthscale in [2.5e-2, inf) --> log(inverse_squared_lengthscales) in
        # (-inf, -2 * log(2.5e-2)], kernel_scale == 1.0 --> log(kernel_scale) == 0.0, and
        # the lower bound of noise_var is handled by `minimum_noise` instead.
        (
            prior.HVARFNER_KERNEL_PARAM_BOUNDS,
            [-math.inf, -math.inf, 0.0, -math.inf],
            [-2.0 * math.log(2.5e-2), -2.0 * math.log(2.5e-2), 0.0, math.inf],
        ),
        (
            prior.KernelParamBounds(lengthscale=(0.5, 2.0), noise_var=(0.0, 1.1)),
            [-2.0 * math.log(2.0), -2.0 * math.log(2.0), -math.inf, -math.inf],
            [-2.0 * math.log(0.5), -2.0 * math.log(0.5), math.inf, math.log(1.0)],
        ),
    ],
)
def test_get_raw_param_bounds(
    bounds: prior.KernelParamBounds, ans_lower: list[float], ans_upper: list[float]
) -> None:
    lower, upper = _get_raw_param_bounds(bounds, n_params=2, minimum_noise=0.1)
    assert np.allclose(lower, ans_lower) and np.allclose(upper, ans_upper)


def _fit_with_bounds(
    bounds: prior.KernelParamBounds,
    kernel_type: KernelChoiceType,
    deterministic_objective: bool = False,
) -> GPRegressor:
    rng = np.random.RandomState(0)
    X = rng.random(size=(20, 4))
    Y = rng.randn(20)
    return fit_kernel_params(
        X=X,
        Y=(Y - Y.mean()) / Y.std(),
        is_categorical=np.zeros(X.shape[1], dtype=bool),
        log_prior=prior.hvarfner_log_prior,
        minimum_noise=prior.DEFAULT_MINIMUM_NOISE_VAR,
        deterministic_objective=deterministic_objective,
        kernel_type=kernel_type,
        kernel_param_bounds=bounds,
    )


@pytest.mark.parametrize("kernel_type", ["matern", "rbf"])
@pytest.mark.parametrize("deterministic_objective", [True, False])
def test_fit_kernel_params_respects_hvarfner_bounds(
    kernel_type: KernelChoiceType, deterministic_objective: bool
) -> None:
    gpr = _fit_with_bounds(
        prior.HVARFNER_KERNEL_PARAM_BOUNDS, kernel_type, deterministic_objective
    )
    lengthscale_min, _ = prior.HVARFNER_KERNEL_PARAM_BOUNDS.lengthscale
    noise_var_min, _ = prior.HVARFNER_KERNEL_PARAM_BOUNDS.noise_var
    assert np.all(gpr.length_scales >= lengthscale_min)
    assert gpr.noise_var.item() >= noise_var_min
    assert gpr.kernel_scale.item() == pytest.approx(1.0), "kernel_scale must be fixed to 1.0."


@pytest.mark.parametrize("kernel_type", ["matern", "rbf"])
def test_fit_kernel_params_with_fixed_bounds(kernel_type: KernelChoiceType) -> None:
    # `min == max` must fix the parameters, which verifies that L-BFGS-B honors the bounds even
    # when the loss gradient points outside of them.
    bounds = prior.KernelParamBounds(
        lengthscale=(0.5, 0.5), kernel_scale=(2.0, 2.0), noise_var=(0.25, 0.5)
    )
    gpr = _fit_with_bounds(bounds, kernel_type)
    assert np.allclose(gpr.length_scales, 0.5)
    assert gpr.kernel_scale.item() == pytest.approx(2.0)
    assert 0.25 <= gpr.noise_var.item() <= 0.5


def test_hvarfner_log_prior_peaks_at_the_mode() -> None:
    dim = 5
    # The mode of LogNormal(mu, var) is exp(mu - var), and mu = sqrt(2) + 0.5 * log(dim).
    mode = math.exp(math.sqrt(2.0) + 0.5 * math.log(dim) - 3.0)
    lengthscales = torch.tensor([mode * 0.5, mode, mode * 2.0], dtype=torch.float64)
    log_priors = [
        prior.hvarfner_log_prior(
            GPRegressor(
                is_categorical=torch.zeros(dim, dtype=torch.bool),
                X_train=torch.zeros((1, dim), dtype=torch.float64),
                y_train=torch.zeros(1, dtype=torch.float64),
                inverse_squared_lengthscales=torch.full((dim,), ls**-2, dtype=torch.float64),
                kernel_scale=torch.tensor(1.0, dtype=torch.float64),
                noise_var=torch.tensor(1e-2, dtype=torch.float64),
                kernel_type="rbf",
            )
        ).item()
        for ls in lengthscales
    ]
    assert log_priors[1] > log_priors[0] and log_priors[1] > log_priors[2]
