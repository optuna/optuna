from typing import Any
from typing import Dict
from typing import List

import optuna
from optuna.distributions import BaseDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial


class Step:
    def __init__(self, search_space_fn, sampler, n_trials) -> None:
        self._search_space_fn = search_space_fn
        self._sampler = sampler
        self._n_trials = n_trials

    def get_search_space(self, current_params: Dict[str, Any]) -> Dict[str, BaseDistribution]:
        return self._search_space_fn(current_params)


class StepwiseSampler(optuna.samplers.BaseSampler):
    def __init__(self, steps: List[Step], default_params: Dict[str, Any]) -> None:
        self.steps = steps
        self.default_params = default_params

    def reseed_rng(self) -> None:
        pass

    def _get_step(self, study, trial) -> Step:
        number = trial.number
        cum_steps = 0
        for step in self.steps:
            if cum_steps <= number < cum_steps + step._n_trials:
                return step
            cum_steps += step._n_trials
        study.stop()
        return self.steps[-1]

    def _get_default_params(self, study: Study):
        if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 0:
            return study.best_params
        return self.default_params

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:

        step = self._get_step(study, trial)
        search_space = step.get_search_space(self._get_default_params(study))
        if len(search_space) < 2:
            return {}
        return search_space

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:

        if len(search_space) < 2:
            return {}

        step = self._get_step(study, trial)
        return step._sampler.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        step = self._get_step(study, trial)
        search_space = step.get_search_space(self._get_default_params(study))
        if param_name in search_space:
            return step._sampler.sample_independent(
                study, trial, param_name, search_space[param_name]
            )
        return self._get_default_params(study)[param_name]
