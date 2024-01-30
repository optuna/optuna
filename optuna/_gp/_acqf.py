from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from ._gp import kernel
from ._gp import KernelParams
from ._gp import MATERN_KERNEL0
from ._gp import posterior
from ._search_space import ScaleType
from ._search_space import SearchSpace


def standard_logei(z: torch.Tensor) -> torch.Tensor:
    # E_{x ~ N(0, 1)}[max(0, x+z)]

    # We switch the implementation depending on the value of z to
    # avoid numerical instability.
    small = z < -25

    vals = torch.empty_like(z)
    # Eq. (9) in ref: https://arxiv.org/pdf/2310.20708.pdf
    # NOTE: We do not use the third condition because ours is good enough.
    z_small = z[small]
    z_normal = z[~small]
    sqrt_2pi = math.sqrt(2 * math.pi)
    # First condition
    cdf = 0.5 * torch.special.erfc(-z_normal * math.sqrt(0.5))
    pdf = torch.exp(-0.5 * z_normal**2) * (1 / sqrt_2pi)
    vals[~small] = torch.log(z_normal * cdf + pdf)
    # Second condition
    r = math.sqrt(0.5 * math.pi) * torch.special.erfcx(-z_small * math.sqrt(0.5))
    vals[small] = -0.5 * z_small**2 + torch.log((z_small * r + 1) * (1 / sqrt_2pi))
    return vals


def logei(mean: torch.Tensor, var: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    # Return E_{y ~ N(mean, var)}[max(0, y-f0)]
    sigma = torch.sqrt(var)
    st_val = standard_logei((mean - f0) / sigma)
    val = torch.log(sigma) + st_val
    return val


def eval_logei(
    kernel_params: KernelParams,
    X: torch.Tensor,
    is_categorical: torch.Tensor,
    cov_Y_Y_inv: torch.Tensor,
    cov_Y_Y_inv_Y: torch.Tensor,
    max_Y: torch.Tensor,
    x: torch.Tensor,
    # Additional noise to prevent numerical instability.
    # Usually this is set to a very small value.
    stabilizing_noise: float,
) -> torch.Tensor:
    cov_fx_fX = kernel(is_categorical, kernel_params, x[..., None, :], X)[..., 0, :]
    cov_fx_fx = MATERN_KERNEL0 * kernel_params.kernel_scale
    (mean, var) = posterior(cov_Y_Y_inv, cov_Y_Y_inv_Y, cov_fx_fX, cov_fx_fx)
    val = logei(mean, var + stabilizing_noise, max_Y)

    return val


@dataclass(frozen=True)
class Acqf:
    kernel_params: KernelParams
    X: np.ndarray
    search_space: SearchSpace
    cov_Y_Y_inv: np.ndarray
    cov_Y_Y_inv_Y: np.ndarray
    max_Y: np.ndarray
    acqf_stabilizing_noise: float


def create_acqf(
    kernel_params: KernelParams,
    search_space: SearchSpace,
    X: np.ndarray,
    Y: np.ndarray,
    acqf_stabilizing_noise: float = 1e-12,
) -> Acqf:
    with torch.no_grad():
        K = kernel(
            torch.from_numpy(search_space.scale_types == ScaleType.CATEGORICAL),
            kernel_params,
            torch.from_numpy(X),
            torch.from_numpy(X),
        )
        cov_Y_Y_inv = (
            torch.linalg.inv(K + kernel_params.noise * torch.eye(X.shape[0], dtype=torch.float64))
            .detach()
            .numpy()
        )
    cov_Y_Y_inv_Y = cov_Y_Y_inv @ Y

    return Acqf(
        kernel_params=kernel_params,
        X=X,
        search_space=search_space,
        cov_Y_Y_inv=cov_Y_Y_inv,
        cov_Y_Y_inv_Y=cov_Y_Y_inv_Y,
        max_Y=np.max(Y),
        acqf_stabilizing_noise=acqf_stabilizing_noise,
    )


def eval_acqf(acqf: Acqf, x: torch.Tensor) -> torch.Tensor:
    return eval_logei(
        kernel_params=acqf.kernel_params,
        X=torch.from_numpy(acqf.X),
        is_categorical=torch.from_numpy(acqf.search_space.scale_types == ScaleType.CATEGORICAL),
        cov_Y_Y_inv=torch.from_numpy(acqf.cov_Y_Y_inv),
        cov_Y_Y_inv_Y=torch.from_numpy(acqf.cov_Y_Y_inv_Y),
        max_Y=torch.tensor(acqf.max_Y, dtype=torch.float64),
        x=x,
        stabilizing_noise=acqf.acqf_stabilizing_noise,
    )


def eval_acqf_no_grad(acqf: Acqf, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return eval_acqf(acqf, torch.from_numpy(x)).detach().numpy()
