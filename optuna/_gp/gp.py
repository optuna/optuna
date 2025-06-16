"""Notations in this Gaussian process implementation

X: Observed parameter values with the shape of (len(trials), len(params)).
Y: Observed objective values with the shape of (len(trials), ).
x: (Possibly batched) parameter value(s) to evaluate with the shape of (..., len(params)).
cov_fX_fX: Kernel matrix X = V[f(X)] with the shape of (len(trials), len(trials)).
cov_fx_fX: Kernel matrix Cov[f(x), f(X)] with the shape of (..., len(trials)).
cov_fx_fx: Kernel scalar value x = V[f(x)]. This value is constant for the Matern 5/2 kernel.
cov_Y_Y_inv:
    The inverse of the covariance matrix (V[f(X) + noise])^-1 with the shape of
    (len(trials), len(trials)).
cov_Y_Y_inv_Y: `cov_Y_Y_inv @ Y` with the shape of (len(trials), ).
max_Y: The maximum of Y (Note that we transform the objective values such that it is maximized.)
d2: The squared distance between two points.
is_categorical:
    A boolean array with the shape of (len(params), ). If is_categorical[i] is True, the i-th
    parameter is categorical.
"""

from __future__ import annotations

import math
from typing import Any
from typing import TYPE_CHECKING
import warnings

import numpy as np

from optuna.logging import get_logger


if TYPE_CHECKING:
    from collections.abc import Callable

    import scipy.optimize as so
    import torch
else:
    from optuna._imports import _LazyImport

    so = _LazyImport("scipy.optimize")
    torch = _LazyImport("torch")

logger = get_logger(__name__)


def warn_and_convert_inf(values: np.ndarray) -> np.ndarray:
    is_values_finite = np.isfinite(values)
    if np.all(is_values_finite):
        return values

    warnings.warn("Clip non-finite values to the min/max finite values for GP fittings.")
    is_any_finite = np.any(is_values_finite, axis=0)
    # NOTE(nabenabe): values cannot include nan to apply np.clip properly, but Optuna anyways won't
    # pass nan in values by design.
    return np.clip(
        values,
        np.where(is_any_finite, np.min(np.where(is_values_finite, values, np.inf), axis=0), 0.0),
        np.where(is_any_finite, np.max(np.where(is_values_finite, values, -np.inf), axis=0), 0.0),
    )


class Matern52Kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, squared_distance: torch.Tensor) -> torch.Tensor:
        """
        This method calculates `exp(-sqrt5d) * (1/3 * sqrt5d ** 2 + sqrt5d + 1)` where
        `sqrt5d = sqrt(5 * squared_distance)`.

        Please note that automatic differentiation by PyTorch does not work well at
        `squared_distance = 0` due to zero division, so we manually save the derivative, i.e.,
        `-5/6 * (1 + sqrt5d) * exp(-sqrt5d)`, for the exact derivative calculation.
        """
        sqrt5d = torch.sqrt(5 * squared_distance)
        exp_part = torch.exp(-sqrt5d)
        val = exp_part * ((5 / 3) * squared_distance + sqrt5d + 1)
        # Notice that the derivative is taken w.r.t. d^2, but not w.r.t. d.
        deriv = (-5 / 6) * (sqrt5d + 1) * exp_part
        ctx.save_for_backward(deriv)
        return val

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        # Let x be squared_distance, f(x) be forward(ctx, x), and g(f) be a provided function,
        # then deriv := df/dx, grad := dg/df, and deriv * grad = df/dx * dg/df = dg/dx.
        (deriv,) = ctx.saved_tensors
        return deriv * grad


class GPRegressor:
    def __init__(
        self,
        inverse_squared_lengthscales: torch.Tensor,  # (len(params), )
        kernel_scale: torch.Tensor,  # Scalar
        noise_var: torch.Tensor,  # Scalar
    ) -> None:
        # TODO(nabenabe): Rename the attributes to private with `_`.
        self.inverse_squared_lengthscales = inverse_squared_lengthscales
        self.kernel_scale = kernel_scale
        self.noise_var = noise_var

    def kernel(
        self, is_categorical: torch.Tensor, X1: torch.Tensor, X2: torch.Tensor
    ) -> torch.Tensor:
        """
        Return the kernel matrix with the shape of (..., n_A, n_B) given X1 and X2 each with the
        shapes of (..., n_A, len(params)) and (..., n_B, len(params)).

        If x1 and x2 have the shape of (len(params), ), kernel(x1, x2) is computed as:
            kernel_scale * Matern52Kernel.apply(
                d2(x1, x2) @ inverse_squared_lengthscales
            )
        where if x1[i] is continuous, d2(x1, x2)[i] = (x1[i] - x2[i]) ** 2 and if x1[i] is
        categorical, d2(x1, x2)[i] = int(x1[i] != x2[i]).
        Note that the distance for categorical parameters is the Hamming distance.
        """
        d2 = (X1[..., :, None, :] - X2[..., None, :, :]) ** 2
        d2[..., is_categorical] = (d2[..., is_categorical] > 0.0).type(torch.float64)
        d2 = (d2 * self.inverse_squared_lengthscales).sum(dim=-1)
        return Matern52Kernel.apply(d2) * self.kernel_scale  # type: ignore

    def posterior(
        self,
        X: torch.Tensor,
        is_categorical: torch.Tensor,
        cov_Y_Y_inv: torch.Tensor,
        cov_Y_Y_inv_Y: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # (mean: (...,), var: (...,))
        cov_fx_fX = self.kernel(is_categorical, x[..., None, :], X)[..., 0, :]
        cov_fx_fx = self.kernel_scale  # kernel(x, x) = kernel_scale

        # mean = cov_fx_fX @ inv(cov_fX_fX + noise * I) @ Y
        # var = cov_fx_fx - cov_fx_fX @ inv(cov_fX_fX + noise * I) @ cov_fx_fX.T
        # The shape of both mean and var is (..., ).
        mean = cov_fx_fX @ cov_Y_Y_inv_Y
        var = cov_fx_fx - (cov_fx_fX * (cov_fx_fX @ cov_Y_Y_inv)).sum(dim=-1)
        # We need to clamp the variance to avoid negative values due to numerical errors.
        return mean, torch.clamp(var, min=0.0)

    def marginal_log_likelihood(
        self, X: torch.Tensor, Y: torch.Tensor, is_categorical: torch.Tensor
    ) -> torch.Tensor:  # Scalar
        # -0.5 * log((2pi)^n |C|) - 0.5 * Y^T C^-1 Y, where C^-1 = cov_Y_Y_inv
        # We apply the cholesky decomposition to efficiently compute log(|C|) and C^-1.

        cov_fX_fX = self.kernel(is_categorical, X, X)

        cov_Y_Y_chol = torch.linalg.cholesky(
            cov_fX_fX + self.noise_var * torch.eye(X.shape[0], dtype=torch.float64)
        )
        # log |L| = 0.5 * log|L^T L| = 0.5 * log|C|
        logdet = 2 * torch.log(torch.diag(cov_Y_Y_chol)).sum()
        # cov_Y_Y_chol @ cov_Y_Y_chol_inv_Y = Y --> cov_Y_Y_chol_inv_Y = inv(cov_Y_Y_chol) @ Y
        cov_Y_Y_chol_inv_Y = torch.linalg.solve_triangular(cov_Y_Y_chol, Y[:, None], upper=False)[
            :, 0
        ]
        return -0.5 * (
            logdet
            + X.shape[0] * math.log(2 * math.pi)
            # Y^T C^-1 Y = Y^T inv(L^T L) Y --> cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y
            + (cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y)
        )


def _fit_kernel_params(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
    log_prior: Callable[[GPRegressor], torch.Tensor],
    minimum_noise: float,
    deterministic_objective: bool,
    gpr_cache: GPRegressor,
    gtol: float,
) -> GPRegressor:
    n_params = X.shape[1]

    # We apply log transform to enforce the positivity of the kernel parameters.
    # Note that we cannot just use the constraint because of the numerical unstability
    # of the marginal log likelihood.
    # We also enforce the noise parameter to be greater than `minimum_noise` to avoid
    # pathological behavior of maximum likelihood estimation.
    initial_raw_params = np.concatenate(
        [
            np.log(gpr_cache.inverse_squared_lengthscales.detach().numpy()),
            [
                np.log(gpr_cache.kernel_scale.item()),
                # We add 0.01 * minimum_noise to initial noise_var to avoid instability.
                np.log(gpr_cache.noise_var.item() - 0.99 * minimum_noise),
            ],
        ]
    )

    def loss_func(raw_params: np.ndarray) -> tuple[float, np.ndarray]:
        raw_params_tensor = torch.from_numpy(raw_params)
        raw_params_tensor.requires_grad_(True)
        with torch.enable_grad():  # type: ignore[no-untyped-call]
            gpr = GPRegressor(
                inverse_squared_lengthscales=torch.exp(raw_params_tensor[:n_params]),
                kernel_scale=torch.exp(raw_params_tensor[n_params]),
                noise_var=(
                    torch.tensor(minimum_noise, dtype=torch.float64)
                    if deterministic_objective
                    else torch.exp(raw_params_tensor[n_params + 1]) + minimum_noise
                ),
            )
            loss = -gpr.marginal_log_likelihood(
                torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(is_categorical)
            ) - log_prior(gpr)
            loss.backward()  # type: ignore
            # scipy.minimize requires all the gradients to be zero for termination.
            raw_noise_var_grad = raw_params_tensor.grad[n_params + 1]  # type: ignore
            assert not deterministic_objective or raw_noise_var_grad == 0
        return loss.item(), raw_params_tensor.grad.detach().numpy()  # type: ignore

    # jac=True means loss_func returns the gradient for gradient descent.
    res = so.minimize(
        # Too small `gtol` causes instability in loss_func optimization.
        loss_func,
        initial_raw_params,
        jac=True,
        method="l-bfgs-b",
        options={"gtol": gtol},
    )
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    raw_params_opt_tensor = torch.from_numpy(res.x)

    return GPRegressor(
        inverse_squared_lengthscales=torch.exp(raw_params_opt_tensor[:n_params]),
        kernel_scale=torch.exp(raw_params_opt_tensor[n_params]),
        noise_var=(
            torch.tensor(minimum_noise, dtype=torch.float64)
            if deterministic_objective
            else minimum_noise + torch.exp(raw_params_opt_tensor[n_params + 1])
        ),
    )


def fit_kernel_params(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
    log_prior: Callable[[GPRegressor], torch.Tensor],
    minimum_noise: float,
    deterministic_objective: bool,
    gpr_cache: GPRegressor | None = None,
    gtol: float = 1e-2,
) -> GPRegressor:
    default_gpr_cache = GPRegressor(
        inverse_squared_lengthscales=torch.ones(X.shape[1], dtype=torch.float64),
        kernel_scale=torch.tensor(1.0, dtype=torch.float64),
        noise_var=torch.tensor(1.0, dtype=torch.float64),
    )
    if gpr_cache is None:
        gpr_cache = default_gpr_cache

    error = None
    # First try optimizing the kernel params with the provided kernel parameters in gpr_cache,
    # but if it fails, rerun the optimization with the default kernel parameters above.
    # This increases the robustness of the optimization.
    for gpr_cache_to_use in [gpr_cache, default_gpr_cache]:
        try:
            return _fit_kernel_params(
                X=X,
                Y=Y,
                is_categorical=is_categorical,
                log_prior=log_prior,
                minimum_noise=minimum_noise,
                gpr_cache=gpr_cache_to_use,
                deterministic_objective=deterministic_objective,
                gtol=gtol,
            )
        except RuntimeError as e:
            error = e

    logger.warning(
        f"The optimization of kernel parameters failed: \n{error}\n"
        "The default initial kernel parameters will be used instead."
    )
    return default_gpr_cache
