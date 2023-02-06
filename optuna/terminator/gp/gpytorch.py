from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import gpytorch
import numpy as np
import torch
from torch.optim import Adam
from torch.quasirandom import SobolEngine

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.terminator import _distribution_is_log
from optuna.terminator.gp.base import BaseGaussianProcess
from optuna.terminator.search_space.intersection import IntersectionSearchSpace
from optuna.trial._frozen import FrozenTrial


class GPyTorchModel(gpytorch.models.exact_gp.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
    ) -> None:
        super(GPyTorchModel, self).__init__(train_x, train_y, likelihood)

        assert len(train_x.shape) == 2
        assert len(train_y.shape) == 1
        assert train_x.shape[0] == train_y.shape[0]

        self._mean_module = gpytorch.means.ConstantMean()
        self._covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=train_x.shape[1],
            ),
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self._mean_module(x)
        covar_x = self._covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _train_gpytorch_model(
    model: GPyTorchModel,
    likelihood: gpytorch.likelihoods.Likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
) -> None:
    hypers = {}
    hypers["_covar_module.base_kernel.lengthscale"] = 1.0
    model.initialize(**hypers)

    optimizer = Adam(model.parameters(), lr=0.1)

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()
    for _ in range(500):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()


def _predict_gpytorch_model(
    model: GPyTorchModel,
    likelihood: gpytorch.likelihoods.Likelihood,
    x: torch.Tensor,
) -> gpytorch.distributions.MultivariateNormal:
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        return model(x)


class GPyTorchGaussianProcess(BaseGaussianProcess):
    def __init__(
        self,
    ) -> None:
        self._trials: Optional[List[FrozenTrial]] = None
        self._gamma: Optional[float] = None
        self._t: Optional[float] = None
        self._model: Optional[GPyTorchModel] = None
        self._likelihood = None
        self._x_scaler = _XScaler()
        self._y_scaler = _YScaler()

    def fit(
        self,
        trials: List[FrozenTrial],
    ) -> None:
        self._trials = trials

        x = self._preprocess_x(trials, fit_x_scaler=True)
        self._t = x.shape[0]
        self._gamma = x.shape[1]

        y = np.array([trial.value for trial in trials], dtype=np.float32)

        self._y_scaler = _YScaler()
        self._y_scaler.fit(y)
        y = self._y_scaler.transform(y)

        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()

        x_tensor = torch.tensor(x)
        y_tensor = torch.tensor(y)

        self._model = GPyTorchModel(x_tensor, y_tensor, self._likelihood)

        _train_gpytorch_model(
            model=self._model,
            likelihood=self._likelihood,
            train_x=x_tensor,
            train_y=y_tensor,
        )

    def mean_std(
        self,
        trials: List[FrozenTrial],
    ) -> Tuple[List[float], List[float]]:
        x = self._preprocess_x(trials)

        x_tensor = torch.tensor(x)
        mean_ternsor, std_tensor = self._mean_std_torch(x_tensor)

        mean = mean_ternsor.detach().numpy().tolist()
        std = std_tensor.detach().numpy().tolist()

        mean = self._y_scaler.untransform(mean)
        std = self._y_scaler.untransform_std(std)

        return mean.tolist(), std.tolist()

    def _mean_std_torch(self, x_torch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._model is not None
        distribution = _predict_gpytorch_model(self._model, self._likelihood, x_torch)

        mean = distribution.mean
        _, upper = distribution.confidence_region()
        std = torch.abs(upper - mean) / 2

        return mean, std

    def min_ucb(self) -> float:
        assert self._trials is not None

        mean, std = self.mean_std(self._trials)
        upper = np.array(mean) + np.array(std) * np.sqrt(self.beta())

        return min(upper)

    def min_lcb(self, n_additional_candidates: int = 2000) -> float:
        assert self._trials is not None

        sobol = SobolEngine(self.gamma(), scramble=True)
        x = sobol.draw(n_additional_candidates)

        # Note that x is assumed to be scaled b/w 0-1 to be stacked with the sobol samples.
        x_observed = torch.tensor(self._preprocess_x(self._trials))

        x = torch.vstack([x, x_observed])

        mean_tensor, std_tensor = self._mean_std_torch(x)
        mean = mean_tensor.detach().numpy()
        std = std_tensor.detach().numpy()

        mean = self._y_scaler.untransform(mean)
        std = self._y_scaler.untransform_std(std)

        lower = np.array(mean) - np.array(std) * np.sqrt(self.beta())

        return min(lower)

    def gamma(self) -> float:
        assert self._gamma is not None, "The GP model has not been trained."
        return self._gamma

    def t(self) -> float:
        assert self._t is not None, "The GP model has not been trained."
        return self._t

    def _preprocess_x(
        self,
        trials: List[FrozenTrial],
        fit_x_scaler: bool = False,
    ) -> np.ndarray:
        search_space = IntersectionSearchSpace().calculate(trials)
        x = _OneToHot(search_space).to_x(trials)

        if fit_x_scaler:
            self._x_scaler.fit(x)
        x = self._x_scaler.transfrom(x)

        return x


class _XScaler:
    def __init__(self) -> None:
        self._min_values = None
        self._max_values = None

    def fit(self, x: np.ndarray) -> None:
        self._min_values = np.min(x, axis=0)
        self._max_values = np.max(x, axis=0)

    def transfrom(self, x: np.ndarray) -> np.ndarray:
        assert self._min_values is not None
        assert len(x.shape) == 2

        x_scaled = []
        for x_row in x:
            denominator = self._max_values - self._min_values
            denominator = np.where(
                denominator == 0.0, 1.0, denominator
            )  # This is to avoid zero div.
            x_row = (x_row - self._min_values) / denominator
            x_scaled.append(x_row)

        return np.array(x_scaled)


class _YScaler:
    def __init__(self) -> None:
        self._mean: Optional[float] = None
        self._std: Optional[float] = None

    def fit(self, y: np.ndarray) -> None:
        mean = float(np.mean(y))
        std = float(np.std(y - mean))
        if std == 0.0:
            # Std scaling is unnecessary when the all elements of y is the same.
            std = 1.0

        self._mean = mean
        self._std = std

    def transform(self, y: np.ndarray) -> np.ndarray:
        assert self._mean is not None
        assert self._std is not None

        transformed_y = (np.array(y) - self._mean) / self._std

        return transformed_y

    def untransform(self, y: np.ndarray) -> np.ndarray:
        assert self._mean is not None
        assert self._std is not None

        untransformed_y = np.array(y) * self._std + self._mean

        return untransformed_y

    def untransform_std(self, stds: np.ndarray) -> np.ndarray:
        assert self._mean is not None
        assert self._std is not None

        untransformed_stds = np.array(stds) * self._std

        return untransformed_stds


class _OneToHot:
    def __init__(self, search_space: Dict[str, BaseDistribution]) -> None:
        assert isinstance(search_space, OrderedDict)
        # TODO(g-votte): assert that the search space is intersection
        self._search_space = search_space

    def to_x(self, trials: List[FrozenTrial]) -> np.ndarray:
        x = []
        for trial in trials:
            x_row = []
            for param, distribution in self._search_space.items():
                # Log distributions are assumed to be unscaled in a previous preprocessing step.
                assert not _distribution_is_log(distribution)

                if isinstance(distribution, CategoricalDistribution):
                    ir = distribution.to_internal_repr(trial.params[param])
                    hot = [1.0 if i == ir else 0.0 for i in range(len(distribution.choices))]
                    x_row += hot
                else:
                    param_value = float(trial.params[param])
                    x_row.append(param_value)

            x.append(x_row)

        return np.array(x, dtype=np.float32)
