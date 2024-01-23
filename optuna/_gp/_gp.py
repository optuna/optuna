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
# K_X_X[len(trials), len(trials)]: kernel matrix of X = V[f(X)]
# K_x_X[(batch_len,) len(trials)]: kernel matrix of x and X = Cov[f(x), f(X)]
# K_x_x: kernel value (scalar) of x = V[f(x)].
#     Since we use a Matern 5/2 kernel, we assume this value to be a constant.
# cov_Y_Y_inv[len(trials), len(trials)]: inv of the covariance matrix of Y = (V[f(X) + noise])^-1
# cov_Y_Y_inv_Y[len(trials)]: cov_Y_Y_inv @ Y
# max_Y: maximum of Y (Note that we transform the objective values such that it is maximized.)


class Matern52KernelFromSqdist(torch.autograd.Function):
    @staticmethod
    def forward(ctx: typing.Any, sqdist: torch.Tensor) -> torch.Tensor:  # type: ignore
        sqrt5d = torch.sqrt(5 * sqdist)
        exp_part = torch.exp(-sqrt5d)
        val = exp_part * ((1 / 3) * sqrt5d * sqrt5d + sqrt5d + 1)
        deriv = (-5 / 6) * (sqrt5d + 1) * exp_part
        ctx.save_for_backward(deriv)
        return val

    @staticmethod
    def backward(ctx: typing.Any, grad: torch.Tensor) -> torch.Tensor:  # type: ignore
        (deriv,) = ctx.saved_tensors
        return deriv * grad


# This is the value of the Matern 5/2 kernel at sqdist=0.
MATERN_KERNEL0 = 1.0


def matern52_kernel_from_sqdist(sqdist: torch.Tensor) -> torch.Tensor:
    # sqrt5d = sqrt(5 * sqdist)
    # exp(sqrt5d) * (1/3 * sqrt5d ** 2 + sqrt5d + 1)
    #
    # We cannot let PyTorch differentiate the above expression because
    # the gradient runs into 0/0 at sqdist=0.
    return Matern52KernelFromSqdist.apply(sqdist)  # type: ignore


@dataclass(frozen=True)
class KernelParams:
    # Kernel parameters to fit.
    inv_sq_lengthscales: torch.Tensor
    kernel_scale: torch.Tensor
    noise: torch.Tensor


def kernel(
    is_categorical: torch.Tensor, kernel_params: KernelParams, X1: torch.Tensor, X2: torch.Tensor
) -> torch.Tensor:
    # kernel(x1, x2) = kernel_scale * matern52_kernel_from_sqdist(d2(x1, x2) * inv_sq_lengthscales)
    # d2(x1, x2) = sum_i d2_i(x1_i, x2_i)
    # d2_i(x1_i, x2_i) = (x1_i - x2_i) ** 2  # if x_i is continuous
    # d2_i(x1_i, x2_i) = 1 if x1_i != x2_i else 0  # if x_i is categorical

    d2 = (X1[..., :, None, :] - X2[..., None, :, :]) ** 2
    d2[..., is_categorical] = torch.where(
        d2[..., is_categorical] > 0.0,
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
    )
    d2 = (d2 * kernel_params.inv_sq_lengthscales).sum(dim=-1)
    return matern52_kernel_from_sqdist(d2) * kernel_params.kernel_scale


def posterior(
    cov_Y_Y_inv: torch.Tensor,
    cov_Y_Y_inv_Y: torch.Tensor,
    K_x_X: torch.Tensor,
    K_x_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # mean = K_x_X @ inv(K_X_X + noise * I) @ Y
    # var = K_x_x - K_x_X @ inv(K_X_X + noise * I) @ K_x_X.T

    mean = K_x_X @ cov_Y_Y_inv_Y  # [batch]
    kalman_gain = K_x_X @ cov_Y_Y_inv  # [batch, N]

    var = K_x_x - (K_x_X * kalman_gain).sum(dim=-1)  # [batch]
    # We need to clamp the variance to avoid negative values due to numerical errors.
    return (mean, torch.clamp(var, min=0.0))


def marginal_log_likelihood(
    X: torch.Tensor,
    Y: torch.Tensor,
    is_categorical: torch.Tensor,
    kernel_params: KernelParams,
) -> torch.Tensor:
    # -0.5 * log(2pi|Σ|) - 0.5 * (Y - μ)^T Σ^-1 (Y - μ)), where μ = 0 and Σ^-1 = cov_Y_Y_inv
    # We apply the cholesky decomposition to efficiently compute log(|Σ|) and Σ^-1.

    K_X_X = kernel(is_categorical, kernel_params, X, X)

    cov_Y_Y_chol = torch.linalg.cholesky(
        K_X_X + kernel_params.noise * torch.eye(X.shape[0], dtype=torch.float64)
    )
    logdet = torch.log(torch.diag(cov_Y_Y_chol)).sum()
    chol_cov_inv_Y = torch.linalg.solve_triangular(cov_Y_Y_chol, Y[:, None], upper=False)[:, 0]
    return -0.5 * (logdet + math.log(2 * math.pi) + torch.vdot(chol_cov_inv_Y, chol_cov_inv_Y))


def fit_kernel_params(
    X: np.ndarray,
    Y: np.ndarray,
    is_categorical: np.ndarray,
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
