from __future__ import annotations

from typing import Optional

import numpy as np
from packaging import version

from optuna._imports import try_import
from optuna.distributions import _is_distribution_log
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.search_space import intersection_search_space
from optuna.terminator.improvement.gp.base import BaseGaussianProcess
from optuna.trial._frozen import FrozenTrial
from optuna.trial._state import TrialState


with try_import() as _imports:
    import botorch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize
    from botorch.models.transforms import Standardize
    import gpytorch
    import torch

    if version.parse(botorch.version.version) < version.parse("0.8.0"):
        from botorch.fit import fit_gpytorch_model as fit_gpytorch_mll
    else:
        from botorch.fit import fit_gpytorch_mll

__all__ = [
    "fit_gpytorch_mll",
    "SingleTaskGP",
    "Normalize",
    "Standardize",
    "gpytorch",
    "torch",
]


class _BoTorchGaussianProcess(BaseGaussianProcess):
    def __init__(self) -> None:
        _imports.check()

        self._gp: Optional[SingleTaskGP] = None

    def fit(
        self,
        trials: list[FrozenTrial],
    ) -> None:
        self._trials = trials

        x, bounds = _convert_trials_to_tensors(trials)

        n_params = x.shape[1]

        y = torch.tensor([trial.value for trial in trials], dtype=torch.float64)
        y = torch.unsqueeze(y, 1)

        self._gp = SingleTaskGP(
            x,
            y,
            input_transform=Normalize(d=n_params, bounds=bounds),
            outcome_transform=Standardize(m=1),
        )

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._gp.likelihood, self._gp)

        fit_gpytorch_mll(mll)

    def predict_mean_std(
        self,
        trials: list[FrozenTrial],
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self._gp is not None

        x, _ = _convert_trials_to_tensors(trials)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self._gp.posterior(x)
            mean = posterior.mean
            variance = posterior.variance
            std = variance.sqrt()

        return mean.detach().numpy().squeeze(-1), std.detach().numpy().squeeze(-1)


def _convert_trials_to_tensors(trials: list[FrozenTrial]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a list of FrozenTrial objects to tensors inputs and bounds.

    This function assumes the following condition for input trials:
    - any categorical param is converted to a float or int one;
    - log is unscaled for any float/int distribution;
    - the state is COMPLETE for any trial;
    - direction is MINIMIZE for any trial.
    """
    search_space = intersection_search_space(trials)
    sorted_params = sorted(search_space.keys())

    x = []
    for trial in trials:
        assert trial.state == TrialState.COMPLETE
        x_row = []
        for param in sorted_params:
            distribution = search_space[param]

            assert not _is_distribution_log(distribution)
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
