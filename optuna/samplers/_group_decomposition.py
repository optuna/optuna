from typing import Any
from typing import Dict
from typing import Optional

from optuna._experimental import experimental
from optuna._search_space_group import SearchSpaceGroup
from optuna.distributions import BaseDistribution
from optuna.samplers._base import BaseSampler
from optuna.samplers._search_space import _GroupDecomposedSearchSpace
from optuna.study import BaseStudy
from optuna.study import Study
from optuna.trial import FrozenTrial


@experimental("2.8.0")
class GroupDecompositionSampler(BaseSampler):
    def __init__(self, base_sampler: BaseSampler):
        self._base_sampler = base_sampler
        self._search_space_group: Optional[SearchSpaceGroup] = None

    def infer_relative_search_space(
        self, study: BaseStudy, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        self._search_space_group = _GroupDecomposedSearchSpace().calculate(study)
        print(self._search_space_group.group)
        search_space = {}
        for sub_space in self._search_space_group.group:
            search_space.update(sub_space)
        return search_space

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        assert self._search_space_group is not None
        params = {}
        for sub_space in self._search_space_group.group:
            params.update(self._base_sampler.sample_relative(study, trial, sub_space))
        return params

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._base_sampler.sample_independent(study, trial, param_name, param_distribution)
