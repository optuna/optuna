from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import optuna
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial


class Step:
    def __init__(
        self,
        search_space_fn: Callable[[Dict[str, Any]], Dict[str, BaseDistribution]],
        sampler_fn: Callable[[Dict[str, Any]], BaseSampler],
        n_trials: int,
    ) -> None:
        self._search_space_fn = search_space_fn
        self._sampler_fn = sampler_fn
        self._sampler: Optional[BaseSampler] = None
        self._n_trials = n_trials

    def get_search_space(self, current_params: Dict[str, Any]) -> Dict[str, BaseDistribution]:
        return self._search_space_fn(current_params)

    def get_sampler(self, current_params: Dict[str, Any]) -> BaseSampler:
        if self._sampler:
            return self._sampler

        self._sampler = self._sampler_fn(current_params)
        return self._sampler

    def update_n_trials(self, steps: List["Step"], trial: FrozenTrial) -> None:
        cum_trials = 0
        for step in steps:
            if step is self:
                break
            cum_trials += step._n_trials
        # Note that trial.number starts from zero.
        self._n_trials = trial.number - cum_trials + 1


class StepwiseSampler(optuna.samplers.BaseSampler):
    def __init__(self, steps: List[Step], default_params: Dict[str, Any]) -> None:
        self.steps = steps
        self.default_params = default_params
        self.stop_flag = False

    def reseed_rng(self) -> None:
        for step in self.steps:
            if step._sampler:
                step._sampler.reseed_rng()

    def _get_step(self, study: optuna.study.Study, trial: FrozenTrial) -> Step:
        number = trial.number
        cum_steps = 0
        ret_step = self.steps[-1]
        for step in self.steps:
            if cum_steps <= number < cum_steps + step._n_trials:
                ret_step = step
                break
            cum_steps += step._n_trials
        if ret_step == self.steps[-1] and number >= cum_steps + ret_step._n_trials - 1:
            self.stop_flag = True
            study.stop()
        return ret_step

    def _get_default_params(self, study: Study) -> Dict[str, Any]:
        if study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
            return study.best_params
        return self.default_params

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:

        step = self._get_step(study, trial)
        search_space = step.get_search_space(self._get_default_params(study))
        return search_space

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:

        step = self._get_step(study, trial)
        sampler = step.get_sampler(self._get_default_params(study))
        value = sampler.sample_relative(study, trial, search_space)
        if study._stop_flag and not self.stop_flag:
            study._stop_flag = False
            step.update_n_trials(self.steps, trial)
        return value

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        step = self._get_step(study, trial)
        search_space = step.get_search_space(self._get_default_params(study))
        if param_name not in search_space:
            return self._get_default_params(study)[param_name]

        sampler = step.get_sampler(self._get_default_params(study))
        value = sampler.sample_independent(study, trial, param_name, search_space[param_name])
        if study._stop_flag and not self.stop_flag:
            study._stop_flag = False
            step.update_n_trials(self.steps, trial)
        return value
