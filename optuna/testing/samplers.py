from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Sequence

import optuna
from optuna.distributions import BaseDistribution


class DeterministicSampler:
    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    def infer_relative_search_space(
        self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
    ) -> dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: "optuna.study.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
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

    def before_trial(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> None:
        pass

    def after_trial(
        self,
        study: "optuna.study.Study",
        trial: "optuna.trial.FrozenTrial",
        state: "optuna.trial.TrialState",
        values: Optional[Sequence[float]],
    ) -> None:
        pass

    def reseed_rng(self) -> None:
        pass


class FirstTrialOnlyRandomSampler(optuna.samplers.RandomSampler):
    def sample_relative(
        self,
        study: "optuna.study.Study",
        trial: "optuna.trial.FrozenTrial",
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, float]:
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
