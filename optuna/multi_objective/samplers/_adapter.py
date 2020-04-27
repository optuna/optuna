from typing import Any
from typing import Dict

from optuna.distributions import BaseDistribution
from optuna import multi_objective
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial


class _MultiObjectiveSamplerAdapter(BaseSampler):
    """Adapter for to :class:`~optuna.multi_objective.samplers.BaseMultiObjectiveSampler`.

    This class implements the :class:`~optuna.samplers.BaseSampler` interface.
    When a method is invoked, the handling will be delegated to the given
    :class:`~optuna.multi_objective.samplers.BaseMultiObjectiveSampler` instance.
    """

    def __init__(self, mo_sampler: "multi_objective.samplers.BaseMultiObjectiveSampler") -> None:
        self._mo_sampler = mo_sampler

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        mo_study = multi_objective.study.MultiObjectiveStudy(study)
        mo_trial = multi_objective.trial.FrozenMultiObjectiveTrial(mo_study.n_objectives, trial)
        return self._mo_sampler.infer_relative_search_space(mo_study, mo_trial)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        mo_study = multi_objective.study.MultiObjectiveStudy(study)
        mo_trial = multi_objective.trial.FrozenMultiObjectiveTrial(mo_study.n_objectives, trial)
        return self._mo_sampler.sample_relative(mo_study, mo_trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        mo_study = multi_objective.study.MultiObjectiveStudy(study)
        mo_trial = multi_objective.trial.FrozenMultiObjectiveTrial(mo_study.n_objectives, trial)
        return self._mo_sampler.sample_independent(
            mo_study, mo_trial, param_name, param_distribution
        )
