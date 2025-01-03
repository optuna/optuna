from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import TYPE_CHECKING

import numpy as np

from optuna._gp.gp import kernel
from optuna._gp.gp import KernelParamsTensor
from optuna._gp.gp import posterior
from optuna._gp.search_space import ScaleType
from optuna._gp.search_space import SearchSpace


if TYPE_CHECKING:
    import torch
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


def _sample_from_normal_sobol(dim: int, n_samples: int, seed: int | None = None) -> torch.Tensor:
    sobol_engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
    # The Sobol sequence in [-1, 1].
    samples = 2.0 * (sobol_engine.draw(n_samples, dtype=torch.float64) - 0.5)
    # Inverse transform to standard normal (values to close to 0/1 result in inf values).
    return torch.erfinv(samples) * float(np.sqrt(2))


def logehvi(
    means: torch.Tensor,  # (..., n_objectives)
    cov: torch.Tensor,  # (..., n_objectives, n_objectives)
    fixed_samples: torch.Tensor,  # (n_qmc_samples, n_objectives)
    non_dominated_lower_bounds: torch.Tensor,  # (n_sub_hyper_rectangles, n_objectives)
    non_dominated_upper_bounds: torch.Tensor,  # (n_sub_hyper_rectangles, n_objectives)
    is_objective_independent: bool = False,
) -> torch.Tensor:  # (..., )
    def _loghvi(
        loss_vals: torch.Tensor,  # (..., n_objectives)
        non_dominated_lower_bounds: torch.Tensor,  # (n_sub_hyper_rectangles, n_objectives)
        non_dominated_upper_bounds: torch.Tensor,  # (n_sub_hyper_rectangles, n_objectives)
    ) -> torch.Tensor:  # shape = (..., )
        # This function calculates Eq. (1) of [Daulton20].
        # TODO(nabenabe): Adapt to Eq. (3) when we support batch optimization.
        diff = torch.nn.functional.relu(
            non_dominated_upper_bounds
            - torch.maximum(loss_vals[..., torch.newaxis, :], non_dominated_lower_bounds)
        )
        return torch.special.logsumexp(diff.log().sum(axis=-1), axis=-1)

    def _logehvi(
        loss_vals: torch.Tensor,  # (..., n_qmc_samples, n_objectives)
        non_dominated_lower_bounds: torch.Tensor,  # (n_sub_hyper_rectangles, n_objectives)
        non_dominated_upper_bounds: torch.Tensor,  # (n_sub_hyper_rectangles, n_objectives)
    ) -> torch.Tensor:  # shape = (..., )
        log_n_qmc_samples = float(np.log(loss_vals.shape[-2]))
        log_hvi_vals = _loghvi(loss_vals, non_dominated_lower_bounds, non_dominated_upper_bounds)
        return -log_n_qmc_samples + torch.special.logsumexp(log_hvi_vals, axis=-1)

    # NOTE(nabenabe): By using fixed samples from the Sobol sequence, EHVI becomes deterministic,
    # making it possible to optimize the acqf by l-BFGS.
    n_objectives = means.shape[-1]
    assert cov.shape[-1] == n_objectives and cov.shape[-2] == n_objectives
    if is_objective_independent:
        L = torch.zeros_like(cov, dtype=torch.float64)
        objective_indices = torch.arange(n_objectives)
        L[..., objective_indices, objective_indices] = torch.sqrt(
            cov[..., objective_indices, objective_indices]
        )
    else:
        L = torch.linalg.cholesky(cov)

    # NOTE(nabenabe): matmul (+squeeze) below is equivalent to einsum("BMM,NM->BNM").
    loss_vals_from_qmc = (
        means[..., torch.newaxis, :]
        + torch.matmul(L[..., torch.newaxis, :, :], fixed_samples[..., torch.newaxis]).squeeze()
    )
    return _logehvi(loss_vals_from_qmc, non_dominated_lower_bounds, non_dominated_upper_bounds)


def standard_logei(z: torch.Tensor) -> torch.Tensor:
    # Return E_{x ~ N(0, 1)}[max(0, x+z)]

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


def logei(mean: torch.Tensor, var: torch.Tensor, f0: float) -> torch.Tensor:
    # Return E_{y ~ N(mean, var)}[max(0, y-f0)]
    sigma = torch.sqrt(var)
    st_val = standard_logei((mean - f0) / sigma)
    val = torch.log(sigma) + st_val
    return val


def logpi(mean: torch.Tensor, var: torch.Tensor, f0: float) -> torch.Tensor:
    # Return the integral of N(mean, var) from -inf to f0
    # This is identical to the integral of N(0, 1) from -inf to (f0-mean)/sigma
    # Return E_{y ~ N(mean, var)}[bool(y <= f0)]
    sigma = torch.sqrt(var)
    return torch.special.log_ndtr((f0 - mean) / sigma)


def ucb(mean: torch.Tensor, var: torch.Tensor, beta: float) -> torch.Tensor:
    return mean + torch.sqrt(beta * var)


def lcb(mean: torch.Tensor, var: torch.Tensor, beta: float) -> torch.Tensor:
    return mean - torch.sqrt(beta * var)


# TODO(contramundum53): consider abstraction for acquisition functions.
# NOTE: Acquisition function is not class on purpose to integrate numba in the future.
class AcquisitionFunctionType(IntEnum):
    LOG_EI = 0
    UCB = 1
    LCB = 2
    LOG_PI = 3


@dataclass(frozen=True)
class AcquisitionFunctionParams:
    acqf_type: AcquisitionFunctionType
    kernel_params: KernelParamsTensor
    X: np.ndarray
    search_space: SearchSpace
    cov_Y_Y_inv: np.ndarray
    cov_Y_Y_inv_Y: np.ndarray
    # TODO(kAIto47802): Want to change the name to a generic name like threshold,
    # since it is not actually in operation as max_Y
    max_Y: float
    beta: float | None
    acqf_stabilizing_noise: float


@dataclass(frozen=True)
class ConstrainedAcquisitionFunctionParams(AcquisitionFunctionParams):
    acqf_params_for_constraints: list[AcquisitionFunctionParams]

    @classmethod
    def from_acqf_params(
        cls,
        acqf_params: AcquisitionFunctionParams,
        acqf_params_for_constraints: list[AcquisitionFunctionParams],
    ) -> ConstrainedAcquisitionFunctionParams:
        return cls(
            acqf_type=acqf_params.acqf_type,
            kernel_params=acqf_params.kernel_params,
            X=acqf_params.X,
            search_space=acqf_params.search_space,
            cov_Y_Y_inv=acqf_params.cov_Y_Y_inv,
            cov_Y_Y_inv_Y=acqf_params.cov_Y_Y_inv_Y,
            max_Y=acqf_params.max_Y,
            beta=acqf_params.beta,
            acqf_stabilizing_noise=acqf_params.acqf_stabilizing_noise,
            acqf_params_for_constraints=acqf_params_for_constraints,
        )


def create_acqf_params(
    acqf_type: AcquisitionFunctionType,
    kernel_params: KernelParamsTensor,
    search_space: SearchSpace,
    X: np.ndarray,
    Y: np.ndarray,
    max_Y: float | None = None,
    beta: float | None = None,
    acqf_stabilizing_noise: float = 1e-12,
) -> AcquisitionFunctionParams:
    X_tensor = torch.from_numpy(X)
    is_categorical = torch.from_numpy(search_space.scale_types == ScaleType.CATEGORICAL)
    with torch.no_grad():
        cov_Y_Y = kernel(is_categorical, kernel_params, X_tensor, X_tensor).detach().numpy()

    cov_Y_Y[np.diag_indices(X.shape[0])] += kernel_params.noise_var.item()
    cov_Y_Y_inv = np.linalg.inv(cov_Y_Y)

    return AcquisitionFunctionParams(
        acqf_type=acqf_type,
        kernel_params=kernel_params,
        X=X,
        search_space=search_space,
        cov_Y_Y_inv=cov_Y_Y_inv,
        cov_Y_Y_inv_Y=cov_Y_Y_inv @ Y,
        max_Y=max_Y if max_Y is not None else np.max(Y),
        beta=beta,
        acqf_stabilizing_noise=acqf_stabilizing_noise,
    )


def eval_acqf(acqf_params: AcquisitionFunctionParams, x: torch.Tensor) -> torch.Tensor:
    mean, var = posterior(
        acqf_params.kernel_params,
        torch.from_numpy(acqf_params.X),
        torch.from_numpy(acqf_params.search_space.scale_types == ScaleType.CATEGORICAL),
        torch.from_numpy(acqf_params.cov_Y_Y_inv),
        torch.from_numpy(acqf_params.cov_Y_Y_inv_Y),
        x,
    )

    if acqf_params.acqf_type == AcquisitionFunctionType.LOG_EI:
        # If there are no feasible trials, max_Y is set to -np.inf.
        # If max_Y is set to -np.inf, we set logEI to zero to ignore it.
        f_val = (
            logei(mean=mean, var=var + acqf_params.acqf_stabilizing_noise, f0=acqf_params.max_Y)
            if not np.isneginf(acqf_params.max_Y)
            else torch.tensor(0.0, dtype=torch.float64)
        )
    elif acqf_params.acqf_type == AcquisitionFunctionType.LOG_PI:
        f_val = logpi(
            mean=mean, var=var + acqf_params.acqf_stabilizing_noise, f0=acqf_params.max_Y
        )
    elif acqf_params.acqf_type == AcquisitionFunctionType.UCB:
        assert acqf_params.beta is not None, "beta must be given to UCB."
        f_val = ucb(mean=mean, var=var, beta=acqf_params.beta)
    elif acqf_params.acqf_type == AcquisitionFunctionType.LCB:
        assert acqf_params.beta is not None, "beta must be given to LCB."
        f_val = lcb(mean=mean, var=var, beta=acqf_params.beta)
    else:
        assert False, "Unknown acquisition function type."

    if isinstance(acqf_params, ConstrainedAcquisitionFunctionParams):
        c_val = sum(eval_acqf(params, x) for params in acqf_params.acqf_params_for_constraints)
        return f_val + c_val
    else:
        return f_val


def eval_acqf_no_grad(acqf_params: AcquisitionFunctionParams, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return eval_acqf(acqf_params, torch.from_numpy(x)).detach().numpy()


def eval_acqf_with_grad(
    acqf_params: AcquisitionFunctionParams, x: np.ndarray
) -> tuple[float, np.ndarray]:
    assert x.ndim == 1
    x_tensor = torch.from_numpy(x)
    x_tensor.requires_grad_(True)
    val = eval_acqf(acqf_params, x_tensor)
    val.backward()  # type: ignore
    return val.item(), x_tensor.grad.detach().numpy()  # type: ignore
