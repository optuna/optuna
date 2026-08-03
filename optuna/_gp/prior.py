from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch

    from optuna._gp import gp
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


DEFAULT_MINIMUM_NOISE_VAR = 1e-6


@dataclasses.dataclass(frozen=True)
class KernelParamBounds:
    """Box constraints imposed on the kernel parameters during the MAP estimation.

    Each attribute is a ``(min, max)`` pair on the natural parameter, i.e., not on the internally
    log-transformed one, and ``(0.0, inf)`` means that the parameter is unconstrained.
    Note that ``min == max`` fixes ``lengthscale`` or ``kernel_scale`` to that value, while
    ``noise_var`` cannot be fixed this way because its lower bound is enforced by ``minimum_noise``
    of :func:`~optuna._gp.gp.fit_kernel_params`, which ``noise_var`` never attains. Use
    ``deterministic_objective=True`` to fix ``noise_var`` instead.

    These bounds exist because some priors, e.g. :func:`hvarfner_log_prior`, are defined only on a
    truncated domain, so the constraints must be handled by the optimizer (L-BFGS-B) rather than by
    the prior itself.
    """

    lengthscale: tuple[float, float] = (0.0, math.inf)
    kernel_scale: tuple[float, float] = (0.0, math.inf)
    noise_var: tuple[float, float] = (0.0, math.inf)

    def __post_init__(self) -> None:
        for name in ("lengthscale", "kernel_scale", "noise_var"):
            param_min, param_max = getattr(self, name)
            assert 0.0 <= param_min <= param_max, (
                f"Got an invalid bound {name}=({param_min}, {param_max})."
            )


DEFAULT_KERNEL_PARAM_BOUNDS = KernelParamBounds()

# Ref. C. Hvarfner et al., Vanilla Bayesian Optimization Performs Great in High Dimensions.
# The lengthscale and noise_var bounds are those used in the paper, and kernel_scale is fixed to
# 1.0 because `hvarfner_log_prior` does not include it (the objective values are standardized).
HVARFNER_KERNEL_PARAM_BOUNDS = KernelParamBounds(
    lengthscale=(2.5e-2, math.inf), kernel_scale=(1.0, 1.0), noise_var=(1e-4, math.inf)
)


def default_log_prior(gpr: gp.GPRegressor) -> torch.Tensor:
    # Log of prior distribution of kernel parameters.

    def gamma_log_prior(x: torch.Tensor, concentration: float, rate: float) -> torch.Tensor:
        # We omit the constant factor `rate ** concentration / Gamma(concentration)`.
        return (concentration - 1) * torch.log(x) - rate * x

    # NOTE(contramundum53): The priors below (params and function
    # shape for inverse_squared_lengthscales) were picked by heuristics.
    # TODO(contramundum53): Check whether these priors are appropriate.
    return (
        -(0.1 / gpr.inverse_squared_lengthscales + 0.1 * gpr.inverse_squared_lengthscales).sum()
        + gamma_log_prior(gpr.kernel_scale, 2, 1)
        + gamma_log_prior(gpr.noise_var, 1.1, 30)
    )


def _log_lognormal_prior(x: torch.Tensor, mu: float, var: float) -> torch.Tensor:
    # NOTE(nabenabe): This prior omits constant factors.
    log_x = torch.log(x)
    return -(log_x**2) / (2.0 * var) + (mu / var - 1) * log_x


def hvarfner_log_prior(gpr: gp.GPRegressor) -> torch.Tensor:
    """Log of the dimension-scaled log-normal prior of the kernel parameters.

    Ref. C. Hvarfner et al., Vanilla Bayesian Optimization Performs Great in High Dimensions.

    Since this prior neither includes ``kernel_scale`` nor is defined outside of the domain assumed
    in the paper, it must be used together with :data:`HVARFNER_KERNEL_PARAM_BOUNDS`.
    """
    lengthscales = torch.rsqrt(gpr.inverse_squared_lengthscales)
    mu = math.sqrt(2.0) + 0.5 * math.log(lengthscales.shape[-1])
    return _log_lognormal_prior(lengthscales, mu=mu, var=3.0).sum() + _log_lognormal_prior(
        gpr.noise_var, mu=-4.0, var=1.0
    )
