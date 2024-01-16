from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from ._gp import kernel
from ._gp import KernelParams
from ._gp import MATERN_KERNEL0
from ._gp import posterior
from ._search_space import ParamType
from ._search_space import SearchSpace


def standard_logei(z: torch.Tensor) -> torch.Tensor:
    # E_{x ~ N(0, 1)}[max(0, x+z)]
    small = z < -25
    cdf = 0.5 * torch.special.erfc(-z * math.sqrt(0.5))
    pdf = torch.exp(-0.5 * z**2) * (1.0 / math.sqrt(2 * math.pi))
    val_normal = torch.log(z * cdf + pdf)
    r = math.sqrt(0.5 * math.pi) * torch.special.erfcx(-z * math.sqrt(0.5))
    val_small = -0.5 * z * z + torch.log((z * r + 1) * (1.0 / math.sqrt(2 * math.pi)))
    return torch.where(small, val_small, val_normal)


def logei(mean: torch.Tensor, var: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
    sigma = torch.sqrt(var)
    st_val = standard_logei((mean - f0) / sigma)
    val = 0.5 * torch.log(var) + st_val
    return val


def eval_logei(
    kernel_params: KernelParams,
    X: torch.Tensor,
    is_categorical: torch.Tensor,
    cov_Y_Y_inv: torch.Tensor,
    cov_Y_Y_inv_Y: torch.Tensor,
    max_Y: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    KxX = kernel(is_categorical, kernel_params, x[..., None, :], X)[..., 0, :]
    Kxx = MATERN_KERNEL0 * kernel_params.kernel_scale
    (mean, var) = posterior(cov_Y_Y_inv, cov_Y_Y_inv_Y, KxX, Kxx)
    val = logei(mean, var + kernel_params.noise, max_Y)

    return val


@dataclass(frozen=True)
class Acqf:
    kernel_params: KernelParams
    X: np.ndarray
    search_space: SearchSpace
    cov_Y_Y_inv: np.ndarray
    cov_Y_Y_inv_Y: np.ndarray
    max_Y: np.ndarray


def create_acqf(
    kernel_params: KernelParams, search_space: SearchSpace, X: np.ndarray, Y: np.ndarray
) -> Acqf:
    with torch.no_grad():
        K = kernel(
            torch.from_numpy(search_space.param_type == ParamType.CATEGORICAL),
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
    )


def eval_acqf(acqf: Acqf, x: torch.Tensor) -> torch.Tensor:
    return eval_logei(
        kernel_params=acqf.kernel_params,
        X=torch.from_numpy(acqf.X),
        is_categorical=torch.from_numpy(acqf.search_space.param_type == ParamType.CATEGORICAL),
        cov_Y_Y_inv=torch.from_numpy(acqf.cov_Y_Y_inv),
        cov_Y_Y_inv_Y=torch.from_numpy(acqf.cov_Y_Y_inv_Y),
        max_Y=torch.tensor(acqf.max_Y, dtype=torch.float64),
        x=x,
    )


def eval_acqf_no_grad(acqf: Acqf, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return eval_acqf(acqf, torch.from_numpy(x)).detach().numpy()
