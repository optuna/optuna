from typing import Any
from typing import Dict
from typing import Optional

import numpy

from optuna import distributions
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial

class PartialFixedSampler(BaseSampler):
     def __init__(self, fixed_params: Dict[str, Any], base_sampler: Optional[BaseSampler]) -> None:
        self._fixed_params = fixed_params
        self._base_sampler = base_sampler

    def reseed_rng(self) -> None:

        self._rng = numpy.random.RandomState()

    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:

        search_space = self._base_sampler.infer_relative_search_space(study, trial)
        
        for param_name in self._fixed_params.keys():
            if name in search_space:
                del search_space[name]

        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:

        return self._base_sampler.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:

        param_value = self._fixed_params.get(param_name)

        if param_value != None:
            return param_value
        else:
            return self._base_sampler.sample_independent(study, trial, param_name, param_distribution)

        