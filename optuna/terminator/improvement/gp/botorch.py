from __future__ import annotations

from typing import Optional

import numpy as np

from optuna._imports import try_import
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.terminator import _distribution_is_log
from optuna.terminator._search_space.intersection import IntersectionSearchSpace
from optuna.terminator.improvement.gp.base import BaseGaussianProcess
from optuna.trial._frozen import FrozenTrial
from optuna.trial._state import TrialState


with try_import() as _imports:
    from botorch.fit import fit_gpytorch_model
    from botorch.models import FixedNoiseGP
    from botorch.models.transforms import Normalize
    from botorch.models.transforms import Standardize
    import gpytorch
    import torch

__all__ = [
    "fit_gpytorch_model",
    "FixedNoiseGP",
    "Normalize",
    "Standardize",
    "gpytorch",
    "torch",
]


class _BoTorchGaussianProcess(BaseGaussianProcess):
    def __init__(self) -> None:
        _imports.check()

        self._n_params: Optional[float] = None
        self._n_trials: Optional[float] = None
        self._gp: Optional[FixedNoiseGP] = None

    def fit(
        self,
        trials: list[FrozenTrial],
    ) -> None:
        self._trials = trials

        x, bounds = _convert_trials_to_tensors(trials)

        self._n_trials = x.shape[0]
        self._n_params = x.shape[1]

        y = torch.tensor([trial.value for trial in trials], dtype=torch.float64)
        y = torch.unsqueeze(y, 1)

        noise = 1e-8 * y.std().item()
        self._gp = FixedNoiseGP(
            x,
            y,
            torch.full_like(y, noise),
            input_transform=Normalize(d=self._n_params, bounds=bounds),
            outcome_transform=Standardize(m=1),
        )

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._gp.likelihood, self._gp)

        fit_gpytorch_model(mll)

    def predict_mean_std(
        self,
        trials: list[FrozenTrial],
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._gp is not None

        x, _ = _convert_trials_to_tensors(trials)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():  # type: ignore[no-untyped-call]
            posterior = self._gp.posterior(x)
            mean = posterior.mean
            variance = posterior.variance
            std = variance.sqrt()

        return mean.detach().numpy(), std.detach().numpy()


def _convert_trials_to_tensors(trials: list[FrozenTrial]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a list of FrozenTrial objects to tensors inputs and bounds.

    This function assumes the following condition for input trials:
    - any categorical param is converted to a float or int one;
    - log is unscaled for any float/int distribution;
    - the state is COMPLETE for any trial;
    - direction is MINIMIZE for any trial.
    """
    search_space = IntersectionSearchSpace().calculate(trials)
    sorted_params = sorted(search_space.keys())

    x = []
    for trial in trials:
        assert trial.state == TrialState.COMPLETE
        x_row = []
        for param in sorted_params:
            distribution = search_space[param]

            assert not _distribution_is_log(distribution)
            assert not isinstance(distribution, CategoricalDistribution)

            param_value = float(trial.params[param])
            x_row.append(param_value)

        x.append(x_row)

    min_bounds = []
    max_bounds = []
    for param, distribution in search_space.items():
        assert isinstance(distribution, (FloatDistribution, IntDistribution))
        min_bounds.append(distribution.low)
        max_bounds.append(distribution.high)
    bounds = [min_bounds, max_bounds]

    return torch.tensor(x, dtype=torch.float64), torch.tensor(bounds, dtype=torch.float64)
