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
from optuna._hypervolume import get_non_dominated_box_bounds
from optuna.study._multi_objective import _is_pareto_front


if TYPE_CHECKING:
    import torch
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


def _sample_from_normal_sobol(dim: int, n_samples: int, seed: int | None) -> torch.Tensor:
    # NOTE(nabenabe): Normal Sobol sampling based on BoTorch.
    # https://github.com/pytorch/botorch/blob/v0.13.0/botorch/sampling/qmc.py#L26-L97
    # https://github.com/pytorch/botorch/blob/v0.13.0/botorch/utils/sampling.py#L109-L138
    sobol_samples = torch.quasirandom.SobolEngine(  # type: ignore[no-untyped-call]
        dimension=dim, scramble=True, seed=seed
    ).draw(n_samples, dtype=torch.float64)
    samples = 2.0 * (sobol_samples - 0.5)  # The Sobol sequence in [-1, 1].
    # Inverse transform to standard normal (values to close to -1 or 1 result in infinity).
    return torch.erfinv(samples) * float(np.sqrt(2))


def logehvi(
    Y_post: torch.Tensor,  # (..., n_qmc_samples, n_objectives)
    non_dominated_box_lower_bounds: torch.Tensor,  # (n_boxes, n_objectives)
    non_dominated_box_upper_bounds: torch.Tensor,  # (n_boxes, n_objectives)
) -> torch.Tensor:  # (..., )
    log_n_qmc_samples = float(np.log(Y_post.shape[-2]))
    # This function calculates Eq. (1) of https://arxiv.org/abs/2006.05078.
    # TODO(nabenabe): Adapt to Eq. (3) when we support batch optimization.
    # TODO(nabenabe): Make the calculation here more numerically stable.
    # cf. https://arxiv.org/abs/2310.20708
    # Check the implementations here:
    # https://github.com/pytorch/botorch/blob/v0.13.0/botorch/utils/safe_math.py
    # https://github.com/pytorch/botorch/blob/v0.13.0/botorch/acquisition/multi_objective/logei.py#L146-L266
    _EPS = torch.tensor(1e-12, dtype=torch.float64)  # NOTE(nabenabe): grad becomes nan when EPS=0.
    diff = torch.maximum(
        _EPS,
        torch.minimum(Y_post[..., torch.newaxis, :], non_dominated_box_upper_bounds)
        - non_dominated_box_lower_bounds,
    )
    # NOTE(nabenabe): logsumexp with dim=-1 is for the HVI calculation and that with dim=-2 is for
    # expectation of the HVIs over the fixed_samples.
    return torch.special.logsumexp(diff.log().sum(dim=-1), dim=(-2, -1)) - log_n_qmc_samples


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
    LOG_EHVI = 4


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


@dataclass(frozen=True)
class MultiObjectiveAcquisitionFunctionParams(AcquisitionFunctionParams):
    acqf_params_for_objectives: list[AcquisitionFunctionParams]
    non_dominated_box_lower_bounds: torch.Tensor
    non_dominated_box_upper_bounds: torch.Tensor
    fixed_samples: torch.Tensor

    @classmethod
    def from_acqf_params(
        cls,
        acqf_params_for_objectives: list[AcquisitionFunctionParams],
        Y: np.ndarray,
        n_qmc_samples: int,
        qmc_seed: int | None,
    ) -> MultiObjectiveAcquisitionFunctionParams:
        def _get_non_dominated_box_bounds() -> tuple[torch.Tensor, torch.Tensor]:
            loss_vals = -Y  # NOTE(nabenabe): Y is to be maximized, loss_vals is to be minimized.
            pareto_sols = loss_vals[_is_pareto_front(loss_vals, assume_unique_lexsorted=False)]
            ref_point = np.max(loss_vals, axis=0)
            ref_point = np.nextafter(np.maximum(1.1 * ref_point, 0.9 * ref_point), np.inf)
            lbs, ubs = get_non_dominated_box_bounds(pareto_sols, ref_point)
            # NOTE(nabenabe): Flip back the sign to make them compatible with maximization.
            return torch.from_numpy(-ubs), torch.from_numpy(-lbs)

        fixed_samples = _sample_from_normal_sobol(
            dim=Y.shape[-1], n_samples=n_qmc_samples, seed=qmc_seed
        )
        non_dominated_box_lower_bounds, non_dominated_box_upper_bounds = (
            _get_non_dominated_box_bounds()
        )
        # Since all the objectives are equally important, we simply use the mean of
        # inverse of squared mean lengthscales over all the objectives.
        mean_lengthscales = np.mean(
            [
                1
                / np.sqrt(acqf_params.kernel_params.inverse_squared_lengthscales.detach().numpy())
                for acqf_params in acqf_params_for_objectives
            ],
            axis=0,
        )
        dummy_kernel_params = KernelParamsTensor(
            # inverse_squared_lengthscales is used in optim_mixed.py.
            # cf. https://github.com/optuna/optuna/blob/v4.3.0/optuna/_gp/optim_mixed.py#L200-L209
            inverse_squared_lengthscales=torch.from_numpy(1.0 / mean_lengthscales**2),
            # These parameters will not be used anywhere.
            kernel_scale=torch.empty(0),
            noise_var=torch.empty(0),
        )
        repr_acqf_params = acqf_params_for_objectives[0]
        return cls(
            acqf_type=AcquisitionFunctionType.LOG_EHVI,
            X=repr_acqf_params.X,
            search_space=repr_acqf_params.search_space,
            acqf_stabilizing_noise=repr_acqf_params.acqf_stabilizing_noise,
            acqf_params_for_objectives=acqf_params_for_objectives,
            non_dominated_box_lower_bounds=non_dominated_box_lower_bounds,
            non_dominated_box_upper_bounds=non_dominated_box_upper_bounds,
            fixed_samples=fixed_samples,
            kernel_params=dummy_kernel_params,
            # The variables below will not be used anywhere, so we simply set dummy values.
            cov_Y_Y_inv=np.empty(0),
            cov_Y_Y_inv_Y=np.empty(0),
            max_Y=np.nan,
            beta=None,
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


def _eval_ehvi(
    ehvi_acqf_params: MultiObjectiveAcquisitionFunctionParams, x: torch.Tensor
) -> torch.Tensor:
    X = torch.from_numpy(ehvi_acqf_params.X)
    is_categorical = torch.from_numpy(
        ehvi_acqf_params.search_space.scale_types == ScaleType.CATEGORICAL
    )
    Y_post = []
    fixed_samples = ehvi_acqf_params.fixed_samples
    for i, acqf_params in enumerate(ehvi_acqf_params.acqf_params_for_objectives):
        mean, var = posterior(
            kernel_params=acqf_params.kernel_params,
            X=X,
            is_categorical=is_categorical,
            cov_Y_Y_inv=torch.from_numpy(acqf_params.cov_Y_Y_inv),
            cov_Y_Y_inv_Y=torch.from_numpy(acqf_params.cov_Y_Y_inv_Y),
            x=x,
        )
        stdev = torch.sqrt(var + ehvi_acqf_params.acqf_stabilizing_noise)
        # NOTE(nabenabe): By using fixed samples from the Sobol sequence, EHVI becomes
        # deterministic, making it possible to optimize the acqf by l-BFGS.
        # Sobol is better than the standard Monte-Carlo w.r.t. the approximation stability.
        # cf. Appendix D of https://arxiv.org/pdf/2006.05078
        Y_post.append(mean[..., torch.newaxis] + stdev[..., torch.newaxis] * fixed_samples[..., i])

    # NOTE(nabenabe): Use the following once multi-task GP is supported.
    # L = torch.linalg.cholesky(cov)
    # Y_post = means[..., torch.newaxis, :] + torch.einsum("...MM,SM->...SM", L, fixed_samples)
    return logehvi(
        Y_post=torch.stack(Y_post, dim=-1),
        non_dominated_box_lower_bounds=ehvi_acqf_params.non_dominated_box_lower_bounds,
        non_dominated_box_upper_bounds=ehvi_acqf_params.non_dominated_box_upper_bounds,
    )


def eval_acqf(acqf_params: AcquisitionFunctionParams, x: torch.Tensor) -> torch.Tensor:
    if acqf_params.acqf_type == AcquisitionFunctionType.LOG_EHVI:
        assert isinstance(acqf_params, MultiObjectiveAcquisitionFunctionParams)
        return _eval_ehvi(ehvi_acqf_params=acqf_params, x=x)

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
