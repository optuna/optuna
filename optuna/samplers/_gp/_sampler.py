import optuna
import torch
from typing import Any, Callable, Sequence
import numpy as np
from ._search_space import get_search_space_and_transformed_params, get_untransformed_param, CATEGORICAL
from ._gp import KernelParams, fit_kernel_params
from ._optim._sample import optimize_acqf_sample
from ._acqf import create_acqf

def default_log_prior(kernel_params: KernelParams) -> torch.Tensor:
    def gamma_log_prior(x: torch.Tensor, concentration: float, rate: float) -> torch.Tensor:
        return ((concentration - 1) * torch.log(x) - rate * x)
    
    return (
        gamma_log_prior(kernel_params.inv_sq_lengthscales, 2, 0.5).sum()
        + gamma_log_prior(kernel_params.kernel_scale, 2, 1) 
        + gamma_log_prior(kernel_params.noise, 1.1, 20)
    )

class GPSampler(optuna.samplers.BaseSampler):
    def __init__(
        self, 
        *, 
        seed: int | None = None, 
        independent_sampler: optuna.samplers.BaseSampler | None = None,
        n_startup_trials: int = 10,
        log_prior: Callable[[KernelParams], torch.Tensor] | None = None,
        minimum_noise: float = 1e-6,
    ) -> None:
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._intersection_search_space = optuna.search_space.IntersectionSearchSpace()
        self._n_startup_trials = n_startup_trials
        self._log_prior = log_prior or default_log_prior
        self._last_kernel_params: KernelParams | None = None
        self._minimum_noise = minimum_noise

    def reseed_rng(self) -> None:
        self._independent_sampler.reseed_rng()

    def infer_relative_search_space(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> dict[str, optuna.distributions.BaseDistribution]:
        return self._intersection_search_space.calculate(study)

    def sample_relative(
        self, study: optuna.Study, trial: optuna.trial.FrozenTrial, search_space: dict[str, optuna.distributions.BaseDistribution]
    ) -> dict[str, Any]:
        if search_space == {}:
            return {}

        states = (optuna.trial.TrialState.COMPLETE,)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=True)
        if len(trials) < self._n_startup_trials:
            return {}
        
        

        internal_search_space, transformed_params = get_search_space_and_transformed_params(trials, search_space)
        values = np.array([trial.value for trial in trials])
        if study.direction == optuna.study.StudyDirection.MINIMIZE:
            values = -values
        values -= values.mean()
        values /= values.std()

        kernel_params = fit_kernel_params(
            X=transformed_params, 
            Y=values, 
            is_categorical=internal_search_space.param_type == CATEGORICAL, 
            log_prior=self._log_prior,
            minimum_noise=self._minimum_noise,
            kernel_params0=self._last_kernel_params,
        )
        self._last_kernel_params = kernel_params

        acqf = create_acqf(
            kernel_params=kernel_params,
            search_space=internal_search_space,
            X=transformed_params,
            Y=values,
        )
        x, _ = optimize_acqf_sample(acqf, n_samples=512)
        return get_untransformed_param(search_space, x)


    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ) -> Any:
        return self._independent_sampler.sample_independent(study, trial, param_name, param_distribution)

    def before_trial(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        self._independent_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
        state: optuna.trial.TrialState,
        values: Sequence[float] | None,
    ) -> None:
        self._independent_sampler.after_trial(study, trial, state, values)
