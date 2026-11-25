from __future__ import annotations

from typing import TYPE_CHECKING

from optuna._gp.acqf import BaseAcquisitionFunc


if TYPE_CHECKING:
    import torch

    from optuna._gp.gp import GPRegressor
    from optuna._gp.search_space import SearchSpace
else:
    from optuna._imports import _LazyImport

    torch = _LazyImport("torch")


class UCB(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        beta: float,
    ) -> None:
        self._gpr = gpr
        self._beta = beta
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._gpr.posterior(x)
        return mean + torch.sqrt(self._beta * var)


class LCB(BaseAcquisitionFunc):
    def __init__(
        self,
        gpr: GPRegressor,
        search_space: SearchSpace,
        beta: float,
    ) -> None:
        self._gpr = gpr
        self._beta = beta
        super().__init__(gpr.length_scales, search_space)

    def eval_acqf(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self._gpr.posterior(x)
        return mean - torch.sqrt(self._beta * var)
