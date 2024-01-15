from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Sequence
from typing import TYPE_CHECKING

import numpy as np

import optuna
from optuna._experimental import experimental_class
from optuna._imports import _LazyImport
from optuna.distributions import BaseDistribution
from optuna.samplers._base import BaseSampler
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    import torch

    import optuna._gp._acqf as _acqf
    import optuna._gp._gp as _gp
    import optuna._gp._optim as _optim
    import optuna._gp._search_space as _search_space
else:
    torch = _LazyImport("torch")
    _search_space = _LazyImport("optuna._gp._search_space")
    _gp = _LazyImport("optuna._gp._gp")
    _optim = _LazyImport("optuna._gp._optim")
    _acqf = _LazyImport("optuna._gp._acqf")


def default_log_prior(kernel_params: "_gp.KernelParams") -> "torch.Tensor":
    def gamma_log_prior(x: "torch.Tensor", concentration: float, rate: float) -> "torch.Tensor":
        return (concentration - 1) * torch.log(x) - rate * x

    return (
        gamma_log_prior(kernel_params.inv_sq_lengthscales, 2, 0.5).sum()
        + gamma_log_prior(kernel_params.kernel_scale, 2, 1)
        + gamma_log_prior(kernel_params.noise, 1.1, 20)
    )


@experimental_class("3.6.0")
class GPSampler(BaseSampler):
    """Sampler using Gaussian process-based Bayesian optimization.

    This sampler fits a Gaussian process (GP) to the objective function and optimizes
    the acquisition function to suggest the next parameters.

    The current implementation uses:
    - Matern kernel with nu=2.5 (differentiable twice)
    - Automatic relevance determination (ARD) for a lengthscale of each parameter
    - Gamma prior for lengthscales, kernel scale, and noise scale
    - Log Expected Improvement (logEI) as the acquisition function
    - Quasi-Monte Carlo (QMC) sampling to optimize the acquisition function

    .. note::
        This sampler requires `scipy` and `pytorch` to be installed.
        You can install these dependencies with `pip install scipy pytorch`.

    Args:
        seed:
            Random seed passed to `independent_sampler`.
            Defaults to None (random seed is used).

        independent_sampler:
            Sampler used for initial sampling (for the first `n_startup_trials` trials)
            and for conditional parameters. Defaults to `None` (a random sampler is used).

        n_startup_trials:
            Number of initial trials. Defaults to 10.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        independent_sampler: BaseSampler | None = None,
        n_startup_trials: int = 10,
    ) -> None:
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._intersection_search_space = optuna.search_space.IntersectionSearchSpace()
        self._n_startup_trials = n_startup_trials
        self._log_prior: "Callable[[_gp.KernelParams], torch.Tensor]" = default_log_prior
        self._minimum_noise: float = 1e-6
        self._last_kernel_params: "_gp.KernelParams | None" = None
        self._optimize_n_samples: int = 2048

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return self._intersection_search_space.calculate(study)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        if len(trials) < self._n_startup_trials:
            return {}

        (
            internal_search_space,
            transformed_params,
        ) = _search_space.get_search_space_and_transformed_params(trials, search_space)
        values = np.array([trial.value for trial in trials])
        if study.direction == StudyDirection.MINIMIZE:
            values = -values
        values -= values.mean()

        EPS = 1e-10
        values /= max(EPS, values.std())

        if self._last_kernel_params is not None and len(
            self._last_kernel_params.inv_sq_lengthscales
        ) != len(internal_search_space.param_type):
            self._last_kernel_params = None

        kernel_params = _gp.fit_kernel_params(
            X=transformed_params,
            Y=values,
            is_categorical=internal_search_space.param_type == _search_space.CATEGORICAL,
            log_prior=self._log_prior,
            minimum_noise=self._minimum_noise,
            kernel_params0=self._last_kernel_params,
        )
        self._last_kernel_params = kernel_params

        acqf = _acqf.create_acqf(
            kernel_params=kernel_params,
            search_space=internal_search_space,
            X=transformed_params,
            Y=values,
        )
        x, _ = _optim.optimize_acqf_sample(acqf, n_samples=self._optimize_n_samples)
        return _search_space.get_untransformed_param(search_space, x)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)
