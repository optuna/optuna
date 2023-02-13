from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from optuna._imports import try_import
from optuna.distributions import CategoricalDistribution
from optuna.terminator import _distribution_is_log
from optuna.terminator.gp.base import BaseMinUcbLcbEstimator
from optuna.terminator.search_space.intersection import IntersectionSearchSpace
from optuna.trial._frozen import FrozenTrial
from optuna.trial._state import TrialState


with try_import() as _imports:
    import botorch
    from botorch.models import SingleTaskGP
    from botorch.optim.fit import fit_gpytorch_torch
    import gpytorch
    import torch
    from torch.quasirandom import SobolEngine

__all__ = ["botorch", "fit_gpytorch_torch", "SingleTaskGP", "gpytorch", "torch", "SobolEngine"]


class BoTorchMinUcbLcbEstimator(BaseMinUcbLcbEstimator):
    def __init__(self, min_lcb_n_additional_candidates: int = 2000) -> None:
        _imports.check()

        self._min_lcb_n_additional_candidates = min_lcb_n_additional_candidates

        self._trials: Optional[List[FrozenTrial]] = None
        self._n_params: Optional[float] = None
        self._n_trials: Optional[float] = None
        self._gp: Optional[SingleTaskGP] = None
        self._likelihood = None
        self._x_scaler = _XScaler()
        self._y_scaler = _YScaler()

    def fit(
        self,
        trials: List[FrozenTrial],
    ) -> None:
        self._trials = trials

        x = _convert_trials_to_x_tensors(trials)
        self._x_scaler.fit(x)
        x = self._x_scaler.transfrom(x)

        self._n_trials = x.shape[0]
        self._n_params = x.shape[1]

        y = torch.tensor([trial.value for trial in trials], dtype=torch.float64)
        self._y_scaler.fit(y)
        y = self._y_scaler.transform(y)
        y = torch.unsqueeze(y, 1)

        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=self._n_params,
            ),
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._gp = SingleTaskGP(x, y, likelihood=likelihood, covar_module=covar_module)

        hypers = {}
        hypers["covar_module.base_kernel.lengthscale"] = 1.0
        self._gp.initialize(**hypers)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._gp.likelihood, self._gp)

        mll.train()
        fit_gpytorch_torch(
            mll, optimizer_cls=torch.optim.Adam, options={"lr": 0.1, "maxiter": 500}
        )
        mll.eval()

    def _mean_std(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._gp is not None

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            distribution = self._gp(x)

        mean = distribution.mean
        _, upper = distribution.confidence_region()
        std = torch.abs(upper - mean) / 2

        mean = self._y_scaler.untransform(mean)
        std = self._y_scaler.untransform_std(std)

        return mean, std

    def min_ucb(self) -> float:
        assert self._trials is not None

        x = _convert_trials_to_x_tensors(self._trials)
        x = self._x_scaler.transfrom(x)

        mean, std = self._mean_std(x)

        upper = mean + std * np.sqrt(self._beta())

        return float(torch.min(upper))

    def min_lcb(self) -> float:
        assert self._trials is not None

        sobol = SobolEngine(self._n_params, scramble=True)  # type: ignore[no-untyped-call]
        x = sobol.draw(self._min_lcb_n_additional_candidates)

        # Note that x is assumed to be scaled b/w 0-1 to be stacked with the sobol samples.
        x_observed = _convert_trials_to_x_tensors(self._trials)
        x_observed = self._x_scaler.transfrom(x_observed)

        x = torch.vstack([x, x_observed])

        mean, std = self._mean_std(x)

        lower = mean - std * np.sqrt(self._beta())
        return float(torch.min(lower))

    def _beta(self, delta: float = 0.1) -> float:
        assert self._n_params is not None
        assert self._n_trials is not None

        beta = 2 * np.log(self._n_params * self._n_trials**2 * np.pi**2 / 6 / delta)

        # The following div is according to the original paper: "We then further scale it down
        # by a factor of 5 as defined in the experiments in Srinivas et al. (2010)"
        beta /= 5

        return beta


# TODO(g-votte): use the `input_transform` option of `SingleTaskGP` instead of this class
class _XScaler:
    def __init__(self) -> None:
        self._min_values = None
        self._max_values = None

    def fit(self, x: torch.Tensor) -> None:
        self._min_values = torch.min(x, dim=0).values
        self._max_values = torch.max(x, dim=0).values

    def transfrom(self, x: torch.Tensor) -> torch.Tensor:
        assert self._min_values is not None
        assert len(x.shape) == 2

        x_scaled = torch.zeros(x.shape, dtype=torch.float64)
        for i, x_row in enumerate(x):
            denominator = self._max_values - self._min_values
            denominator = torch.where(
                denominator == 0.0, 1.0, denominator
            )  # This is to avoid zero div.
            x_row = (x_row - self._min_values) / denominator
            x_scaled[i, :] = x_row

        return x_scaled


class _YScaler:
    def __init__(self) -> None:
        self._mean: Optional[float] = None
        self._std: Optional[float] = None

    def fit(self, y: torch.Tensor) -> None:
        self._mean = float(y.mean())
        self._std = float((y - self._mean).std())
        if self._std == 0.0:
            # Std scaling is unnecessary when the all elements of y is the same.
            self._std = 1.0

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        assert self._mean is not None
        assert self._std is not None

        y -= self._mean
        y /= self._std

        return y

    def untransform(self, y: torch.Tensor) -> torch.Tensor:
        assert self._mean is not None
        assert self._std is not None

        y *= self._std
        y += self._mean

        return y

    def untransform_std(self, stds: torch.Tensor) -> torch.Tensor:
        assert self._mean is not None
        assert self._std is not None

        stds *= self._std

        return stds


def _convert_trials_to_x_tensors(trials: List[FrozenTrial]) -> torch.Tensor:
    """Convert a list of FrozenTrial objects to an input tensor of the Gaussian process.

    This function assumes the following condition for input trials:
    - any categorical param is converted to a float or int one;
    - log is unscaled for any float/int distribution;
    - the trial's state is COMPLETE;
    - direction is minimize.
    """
    search_space = IntersectionSearchSpace().calculate(trials)

    x = []
    for trial in trials:
        assert trial.state == TrialState.COMPLETE
        x_row = []
        for param, distribution in search_space.items():
            assert not _distribution_is_log(distribution)
            assert not isinstance(distribution, CategoricalDistribution)

            param_value = float(trial.params[param])
            x_row.append(param_value)

        x.append(x_row)

    return torch.tensor(x, dtype=torch.float64)
