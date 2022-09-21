from typing import Any
from typing import Dict

import optuna
from optuna.distributions import BaseDistribution


class DeterministicSampler(optuna.samplers.BaseSampler):
    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params

    def infer_relative_search_space(
        self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
    ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: "optuna.study.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        return {}

    def sample_independent(
        self,
        study: "optuna.study.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        param_value = self.params[param_name]
        assert param_distribution._contains(param_distribution.to_internal_repr(param_value))
        return param_value


class FirstTrialOnlyRandomSampler(optuna.samplers.RandomSampler):
    def sample_relative(
        self,
        study: "optuna.study.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, float]:

        if len(study.trials) > 1:
            raise RuntimeError("`FirstTrialOnlyRandomSampler` only works on the first trial.")

        return super(FirstTrialOnlyRandomSampler, self).sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: "optuna.study.Study",
        trial: "optuna.trial.FrozenTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> float:

        if len(study.trials) > 1:
            raise RuntimeError("`FirstTrialOnlyRandomSampler` only works on the first trial.")

        return super(FirstTrialOnlyRandomSampler, self).sample_independent(
            study, trial, param_name, param_distribution
        )
