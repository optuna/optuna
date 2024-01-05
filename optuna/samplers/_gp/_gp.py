import torch
import math
import numpy as np
from typing import NamedTuple, Callable
import scipy.optimize

class Matern52KernelFromSqdist(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sqdist: torch.Tensor) -> torch.Tensor:
        sqrt5d = torch.sqrt(5 * sqdist)
        exp_part = torch.exp(-sqrt5d)
        val = exp_part * ((1/3) * sqrt5d * sqrt5d + sqrt5d + 1)
        deriv = (-5/6) * (sqrt5d + 1) * exp_part
        ctx.save_for_backward(deriv)
        return val
    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        deriv, = ctx.saved_tensors
        return deriv * grad
    

MATERN_KERNEL0 = 1.0
def matern52_kernel_from_sqdist(sqdist: torch.Tensor) -> torch.Tensor:
    # sqrt5d = torch.sqrt(5 * sqdist)
    # return torch.exp(-sqrt5d) * ((1/3) * sqrt5d * sqrt5d + sqrt5d + 1)
    return Matern52KernelFromSqdist.apply(sqdist)

class KernelParams(NamedTuple):
    inv_sq_lengthscales: torch.Tensor
    kernel_scale: torch.Tensor
    noise: torch.Tensor

def kernel(is_categorical: torch.Tensor, kernel_params: KernelParams, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
    d2 = (X1[..., :, None, :] - X2[..., None, :, :]) ** 2
    d2[..., is_categorical] = torch.where(
        d2[..., is_categorical] > 0.0, 
        torch.tensor(1.0, dtype=torch.float64), 
        torch.tensor(0.0, dtype=torch.float64)
    )
    d2 = (d2 * kernel_params.inv_sq_lengthscales).sum(dim=-1)
    return matern52_kernel_from_sqdist(d2) * kernel_params.kernel_scale

def posterior(cov_Y_Y_inv: torch.Tensor, cov_Y_Y_inv_Y: torch.Tensor, KxX: torch.Tensor, Kxx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = KxX @ cov_Y_Y_inv_Y  # [batch]
    kalman_gain = KxX @ cov_Y_Y_inv # [batch, N]

    var = Kxx - (KxX * kalman_gain).sum(dim=-1)  # [batch]
    return (mean, torch.clamp(var, min=0.0))

LOG_2PI = math.log(2 * math.pi)
def marginal_log_likelihood(
    X: torch.Tensor,
    Y: torch.Tensor,
    is_categorical: torch.Tensor,
    kernel_params: KernelParams,
) -> torch.Tensor:
    K_X_X = kernel(is_categorical, kernel_params, X, X)

    cov_Y_Y_chol = torch.linalg.cholesky(K_X_X + kernel_params.noise * torch.eye(X.shape[0], dtype=torch.float64))
    logdet = torch.log(torch.diag(cov_Y_Y_chol)).sum()
    chol_cov_inv_Y = torch.linalg.solve_triangular(cov_Y_Y_chol, Y[:, None], upper=False)[:, 0]
    # return -0.5 * log(2pi|Σ|) - 0.5 * (Y - μ)^T Σ^-1 (Y - μ))
    return -0.5 * (logdet + LOG_2PI + torch.vdot(chol_cov_inv_Y, chol_cov_inv_Y))

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
    repr0 = np.concatenate([
        np.log(kernel_params0.inv_sq_lengthscales.detach().numpy()), 
        [np.log(kernel_params0.kernel_scale.item()), 
            np.log(kernel_params0.noise.item() - minimum_noise)]])
    
    def loss_func(repr: np.ndarray) -> tuple[float, np.ndarray]:
        log_inv_sq_lengthscales = torch.tensor(repr[:X.shape[1]], dtype=torch.float64, requires_grad=True)
        log_kernel_scale = torch.tensor(repr[X.shape[1]], dtype=torch.float64, requires_grad=True)
        log_noise = torch.tensor(repr[X.shape[1] + 1], dtype=torch.float64, requires_grad=True)
        params = KernelParams(
            inv_sq_lengthscales=torch.exp(log_inv_sq_lengthscales),
            kernel_scale=torch.exp(log_kernel_scale),
            noise=torch.exp(log_noise) + minimum_noise,
        )
        loss = -marginal_log_likelihood(torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(is_categorical), params) - log_prior(params)
        loss.backward()
        return loss.item(), np.concatenate([
            log_inv_sq_lengthscales.grad.detach().numpy(), 
            [log_kernel_scale.grad.item(), 
                log_noise.grad.item()]])

    res = scipy.optimize.minimize(loss_func, repr0, jac=True)
    repr_opt = res.x

    return KernelParams(
        inv_sq_lengthscales=torch.exp(torch.tensor(repr_opt[:X.shape[1]], dtype=torch.float64)),
        kernel_scale=torch.exp(torch.tensor(repr_opt[X.shape[1]], dtype=torch.float64)),
        noise=torch.exp(torch.tensor(repr_opt[X.shape[1] + 1], dtype=torch.float64)) + minimum_noise,
    )
