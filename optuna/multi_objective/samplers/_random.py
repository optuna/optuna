from typing import Any
from typing import Dict
from typing import Optional

import optuna
from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna import multi_objective
from optuna.multi_objective.samplers import BaseMoSampler


@experimental("1.4.0")
class RandomMoSampler(BaseMoSampler):
    def __init__(self, seed: Optional[int] = None) -> None:
        self._sampler = optuna.samplers.RandomSampler(seed=seed)

    def infer_relative_search_space(
        self, study: "multi_objective.study.MoStudy", trial: "multi_objective.trial.FrozenMoTrial"
    ):
        # TODO(ohta): Convert `study` and `trial` to single objective versions before passing.
        return self._sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self,
        study: "multi_objective.study.MoStudy",
        trial: "multi_objective.trial.FrozenMoTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        # TODO(ohta): Convert `study` and `trial` to single objective versions before passing.
        return self._sampler.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: "multi_objective.study.MoStudy",
        trial: "multi_objective.trial.FrozenMoTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        # TODO(ohta): Convert `study` and `trial` to single objective versions before passing.
        return self._sampler.sample_independent(study, trial, param_name, param_distribution)
