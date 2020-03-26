from typing import Optional

import optuna
from optuna import mo
from optuna.mo.samplers import BaseMoSampler


class RandomSampler(BaseMoSampler):
    def __init__(self, seed: Optional[int]) -> None:
        self._sampler = optuna.samplers.RandomSampler(seed=seed)

    def infer_relative_search_space(
        self, study: "mo.study.MoStudy", trial: "mo.trial.FrozenMoTrial"
    ):
        # TODO(ohta): Convert `study` and `trial` to single objective versions before passing.
        return self._sampler.infer_relative_search_space(study, trial)

    @abc.abstractmethod
    def sample_relative(
        self,
        study: "mo.study.MoStudy",
        trial: "mo.trial.FrozenMoTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        # TODO(ohta): Convert `study` and `trial` to single objective versions before passing.
        return self._sampler.sample_relative(study, trial, search_space)

    @abc.abstractmethod
    def sample_independent(
        self,
        study: "mo.study.MoStudy",
        trial: "mo.trial.FrozenMoTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        # TODO(ohta): Convert `study` and `trial` to single objective versions before passing.
        return self._sampler.sample_independent(study, trial, param_name, param_distribution)
