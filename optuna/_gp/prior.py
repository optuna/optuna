from __future__ import annotations

from typing import TYPE_CHECKING

from optuna._gp import gp


if TYPE_CHECKING:
    import torch
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


DEFAULT_MINIMUM_NOISE_VAR = 1e-6


def default_log_prior(kernel_params: "gp.KernelParamsTensor") -> "torch.Tensor":
    # Log of prior distribution of kernel parameters.

    def gamma_log_prior(x: "torch.Tensor", concentration: float, rate: float) -> "torch.Tensor":
        # We omit the constant factor `rate ** concentration / factorial(concentration)`.
        return (concentration - 1) * torch.log(x) - rate * x

    # NOTE(contramundum53): The parameters below were picked qualitatively.
    # TODO(contramundum53): Check whether these priors are appropriate.
    return (
        gamma_log_prior(kernel_params.inverse_squared_lengthscales, 2, 0.5).sum()
        + gamma_log_prior(kernel_params.kernel_scale, 2, 1)
        + gamma_log_prior(kernel_params.noise_var, 1.1, 20)
    )
