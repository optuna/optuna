from __future__ import annotations

from dataclasses import dataclass
import math
import typing
from typing import Callable

import numpy as np
import scipy.optimize
import torch


# This GP implementation uses the following notation:
# X[len(trials), len(params)]: observed parameter values.
# Y[len(trials)]: observed objective values.
# x[(batch_len,) len(params)]: parameter value to evaluate. Possibly batched.
# cov_fX_fX[len(trials), len(trials)]: kernel matrix of X = V[f(X)]
# cov_fx_fX[(batch_len,) len(trials)]: kernel matrix of x and X = Cov[f(x), f(X)]
# cov_fx_fx: kernel value (scalar) of x = V[f(x)].
#     Since we use a Matern 5/2 kernel, we assume this value to be a constant.
# cov_Y_Y_inv[len(trials), len(trials)]: inv of the covariance matrix of Y = (V[f(X) + noise])^-1
# cov_Y_Y_inv_Y[len(trials)]: cov_Y_Y_inv @ Y
# max_Y: maximum of Y (Note that we transform the objective values such that it is maximized.)
# d2: squared distance between two points


class Matern52Kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx: typing.Any, squared_distance: torch.Tensor) -> torch.Tensor:  # type: ignore
        sqrt5d = torch.sqrt(5 * squared_distance)
        exp_part = torch.exp(-sqrt5d)
        val = exp_part * ((5 / 3) * squared_distance + sqrt5d + 1)
        # Notice that the derivative is taken w.r.t. d^2, but not w.r.t. d.
        deriv = (-5 / 6) * (sqrt5d + 1) * exp_part
        ctx.save_for_backward(deriv)
        return val

    @staticmethod
    def backward(ctx: typing.Any, grad: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Let x be squared_distance, f(x) be forward(ctx, x), and g(f) be a provided function,
        # then deriv := df/dx, grad := dg/df, and deriv * grad = df/dx * dg/df = dg/dx.
        (deriv,) = ctx.saved_tensors
        return deriv * grad


# This is the value of the Matern 5/2 kernel at squared_distance=0.
MATERN_KERNEL0 = 1.0


def matern52_kernel_from_squared_distance(squared_distance: torch.Tensor) -> torch.Tensor:
    # sqrt5d = sqrt(5 * squared_distance)
    # exp(sqrt5d) * (1/3 * sqrt5d ** 2 + sqrt5d + 1)
    #
    # We cannot let PyTorch differentiate the above expression because
    # the gradient runs into 0/0 at squared_distance=0.
    return Matern52Kernel.apply(squared_distance)  # type: ignore


@dataclass(frozen=True)
class KernelParams:
    # Kernel parameters to fit.
    inv_sq_lengthscales: torch.Tensor  # [len(params)]
    kernel_scale: torch.Tensor  # Scalar
    noise: torch.Tensor  # Scalar


def kernel(
    is_categorical: torch.Tensor,  # [len(params)]
    kernel_params: KernelParams,
    X1: torch.Tensor,  # [...batch_shape, n_A, len(params)]
    X2: torch.Tensor,  # [...batch_shape, n_B, len(params)]
) -> torch.Tensor:  # [...batch_shape, n_A, n_B]
    # kernel(x1, x2) = kernel_scale * matern52_kernel_from_squared_distance(
    #                     d2(x1, x2) * inv_sq_lengthscales)
    # d2(x1, x2) = sum_i d2_i(x1_i, x2_i)
    # d2_i(x1_i, x2_i) = (x1_i - x2_i) ** 2  # if x_i is continuous
    # d2_i(x1_i, x2_i) = 1 if x1_i != x2_i else 0  # if x_i is categorical

    d2 = (X1[..., :, None, :] - X2[..., None, :, :]) ** 2

    # Use the Hamming distance for categorical parameters.
    d2[..., is_categorical] = torch.where(
        d2[..., is_categorical] > 0.0,
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
    )
    d2 = (d2 * kernel_params.inv_sq_lengthscales).sum(dim=-1)
    return matern52_kernel_from_squared_distance(d2) * kernel_params.kernel_scale


def posterior(
    cov_Y_Y_inv: torch.Tensor,  # [len(trials), len(trials)]
    cov_Y_Y_inv_Y: torch.Tensor,  # [len(trials)]
    cov_fx_fX: torch.Tensor,  # [(batch,) len(trials)]
    cov_fx_fx: torch.Tensor,  # Scalar or [(batch,)]
) -> tuple[torch.Tensor, torch.Tensor]:  # [(batch,)], [(batch,)]
    # mean = cov_fx_fX @ inv(cov_fX_fX + noise * I) @ Y
    # var = cov_fx_fx - cov_fx_fX @ inv(cov_fX_fX + noise * I) @ cov_fx_fX.T

    mean = cov_fx_fX @ cov_Y_Y_inv_Y  # [batch]
    var = cov_fx_fx - (cov_fx_fX * (cov_fx_fX @ cov_Y_Y_inv)).sum(dim=-1)  # [batch]
    # We need to clamp the variance to avoid negative values due to numerical errors.
    return (mean, torch.clamp(var, min=0.0))


def marginal_log_likelihood(
    X: torch.Tensor,  # [len(trials), len(params)]
    Y: torch.Tensor,  # [len(trials)]
    is_categorical: torch.Tensor,  # [len(params)]
    kernel_params: KernelParams,
) -> torch.Tensor:  # Scalar
    # -0.5 * log(2pi|Σ|) - 0.5 * (Y - μ)^T Σ^-1 (Y - μ)), where μ = 0 and Σ^-1 = cov_Y_Y_inv
    # We apply the cholesky decomposition to efficiently compute log(|Σ|) and Σ^-1.

    cov_fX_fX = kernel(is_categorical, kernel_params, X, X)

    cov_Y_Y_chol = torch.linalg.cholesky(
        cov_fX_fX + kernel_params.noise * torch.eye(X.shape[0], dtype=torch.float64)
    )
    logdet = 2 * torch.log(torch.diag(cov_Y_Y_chol)).sum()  # log |L| = 0.5 * log|L^T L| = 0.5 * log|C| 
    # cov_Y_Y_chol @ cov_Y_Y_chol_inv_Y = Y --> cov_Y_Y_chol_inv_Y = inv(cov_Y_Y_chol) @ Y
    cov_Y_Y_chol_inv_Y = torch.linalg.solve_triangular(cov_Y_Y_chol, Y[:, None], upper=False)[:, 0]
    return -0.5 * (
        logdet
        + X.shape[0] * math.log(2 * math.pi)
        # Y^T C^-1 Y = Y^T inv(L^T L) Y --> cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y
        + (cov_Y_Y_chol_inv_Y @ cov_Y_Y_chol_inv_Y)
    )


def fit_kernel_params(
    X: np.ndarray,  # [len(trials), len(params)]
    Y: np.ndarray,  # [len(trials)]
    is_categorical: np.ndarray,  # [len(params)]
    log_prior: Callable[[KernelParams], torch.Tensor],
    minimum_noise: float = 0.0,
    kernel_params0: KernelParams | None = None,
) -> KernelParams:
    if kernel_params0 is None:
        kernel_params0 = KernelParams(
            inv_sq_lengthscales=torch.ones(X.shape[1], dtype=torch.float64),
            kernel_scale=torch.tensor(1.0, dtype=torch.float64),
            noise=torch.tensor(1.0, dtype=torch.float64),
        )

    # We apply log transform to enforce the positivity of the kernel parameters.
    # Note that we cannot just use the constraint because of the numerical unstability
    # of the marginal log likelihood.
    # We also enforce the noise parameter to be greater than `minimum_noise` to avoid
    # pathological behavior of maximum likelihood estimation.
    repr0 = np.concatenate(
        [
            np.log(kernel_params0.inv_sq_lengthscales.detach().numpy()),
            [
                np.log(kernel_params0.kernel_scale.item()),
                np.log(kernel_params0.noise.item() - minimum_noise),
            ],
        ]
    )

    def loss_func(repr: np.ndarray) -> tuple[float, np.ndarray]:
        log_inv_sq_lengthscales = torch.tensor(
            repr[: X.shape[1]], dtype=torch.float64, requires_grad=True
        )
        log_kernel_scale = torch.tensor(repr[X.shape[1]], dtype=torch.float64, requires_grad=True)
        log_noise = torch.tensor(repr[X.shape[1] + 1], dtype=torch.float64, requires_grad=True)
        params = KernelParams(
            inv_sq_lengthscales=torch.exp(log_inv_sq_lengthscales),
            kernel_scale=torch.exp(log_kernel_scale),
            noise=torch.exp(log_noise) + minimum_noise,
        )
        loss = -marginal_log_likelihood(
            torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(is_categorical), params
        ) - log_prior(params)
        loss.backward()  # type: ignore
        return loss.item(), np.concatenate(
            [
                log_inv_sq_lengthscales.grad.detach().numpy(),  # type: ignore
                [log_kernel_scale.grad.item(), log_noise.grad.item()],  # type: ignore
            ]
        )

    res = scipy.optimize.minimize(loss_func, repr0, jac=True)
    repr_opt = res.x

    return KernelParams(
        inv_sq_lengthscales=torch.exp(torch.tensor(repr_opt[: X.shape[1]], dtype=torch.float64)),
        kernel_scale=torch.exp(torch.tensor(repr_opt[X.shape[1]], dtype=torch.float64)),
        noise=torch.exp(torch.tensor(repr_opt[X.shape[1] + 1], dtype=torch.float64))
        + minimum_noise,
    )
