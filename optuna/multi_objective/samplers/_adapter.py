from typing import Any
from typing import Dict

from optuna import multi_objective
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.structs import FrozenTrial
from optuna.study import Study


class _MoSamplerAdapter(BaseSampler):
    def __init__(self, mo_sampler: "multi_objective.samplers.BaseMoSampler") -> None:
        self._mo_sampler = mo_sampler

    def infer_relative_search_space(self, study: Study, trial: FrozenTrial):
        mo_study = multi_objective.study.MoStudy(study)
        mo_trial = multi_objective.trial.FrozenMoTrial(mo_study.n_objectives, trial)
        return self._mo_sampler.infer_relative_search_space(mo_study, mo_trial)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        mo_study = multi_objective.study.MoStudy(study)
        mo_trial = multi_objective.trial.FrozenMoTrial(mo_study.n_objectives, trial)
        return self._mo_sampler.sample_relative(mo_study, mo_trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        mo_study = multi_objective.study.MoStudy(study)
        mo_trial = multi_objective.trial.FrozenMoTrial(mo_study.n_objectives, trial)
        return self._mo_sampler.sample_independent(
            mo_study, mo_trial, param_name, param_distribution
        )
