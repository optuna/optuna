from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from optuna._imports import try_import
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.terminator import _distribution_is_log
from optuna.terminator.gp.base import BaseMinUcbLcbEstimator
from optuna.terminator.improvement.preprocessing import AddRandomInputs
from optuna.terminator.improvement.preprocessing import OneToHot
from optuna.terminator.improvement.preprocessing import PreprocessingPipeline
from optuna.terminator.search_space.intersection import IntersectionSearchSpace
from optuna.trial._frozen import FrozenTrial
from optuna.trial._state import TrialState


with try_import() as _imports:
    import botorch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize
    from botorch.models.transforms import Standardize
    from botorch.optim.fit import fit_gpytorch_torch
    import gpytorch
    import torch
    from torch.quasirandom import SobolEngine

__all__ = [
    "botorch",
    "SingleTaskGP",
    "Normalize",
    "Standardize",
    "fit_gpytorch_torch",
    "gpytorch",
    "torch",
    "SobolEngine",
]


class BoTorchMinUcbLcbEstimator(BaseMinUcbLcbEstimator):
    def __init__(self, min_lcb_n_additional_candidates: int = 2000) -> None:
        _imports.check()

        self._min_lcb_n_additional_candidates = min_lcb_n_additional_candidates

        self._trials: Optional[List[FrozenTrial]] = None
        self._search_space: Optional[Dict[str, BaseDistribution]] = None
        self._n_params: Optional[float] = None
        self._n_trials: Optional[float] = None
        self._gp: Optional[SingleTaskGP] = None

    def fit(
        self,
        trials: List[FrozenTrial],
    ) -> None:
        self._trials = trials

        # TODO(g-votte): guarantee that _search_space is an ordered dict
        self._search_space = IntersectionSearchSpace().calculate(trials)

        preprocessing = OneToHot()

        trials = preprocessing.apply(self._trials, None)

        x, bounds = _convert_trials_to_tensors(trials)

        self._n_trials = x.shape[0]
        self._n_params = x.shape[1]

        y = torch.tensor([trial.value for trial in trials], dtype=torch.float64)
        y = torch.unsqueeze(y, 1)

        covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=self._n_params,
            ),
        )

        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self._gp = SingleTaskGP(
            x,
            y,
            likelihood=likelihood,
            covar_module=covar_module,
            input_transform=Normalize(d=self._n_params, bounds=bounds),
            outcome_transform=Standardize(m=1),
        )

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
            posterior = self._gp.posterior(x)
            mean = posterior.mean
            variance = posterior.variance
            std = variance.sqrt()

        return mean, std

    def min_ucb(self) -> float:
        assert self._trials is not None

        preprocessing = OneToHot()
        trials = preprocessing.apply(self._trials, None)
        x, _ = _convert_trials_to_tensors(trials)

        mean, std = self._mean_std(x)

        upper = mean + std * np.sqrt(self._beta())

        return float(torch.min(upper))

    def min_lcb(self) -> float:
        assert self._trials is not None

        preprocessing = PreprocessingPipeline(
            [
                AddRandomInputs(
                    self._min_lcb_n_additional_candidates, search_space=self._search_space
                ),
                OneToHot(),
            ]
        )
        trials = preprocessing.apply(self._trials, None)
        x, _ = _convert_trials_to_tensors(trials)

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


def _convert_trials_to_tensors(
    trials: List[FrozenTrial],
) -> Tuple[torch.Tensor]:
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
