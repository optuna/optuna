import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import warnings

import numpy as np
import scipy.special
from scipy.stats import truncnorm

from optuna import distributions
from optuna._study_direction import StudyDirection
from optuna.distributions import BaseDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.logging import get_logger
from optuna.samplers._base import BaseSampler
from optuna.samplers._random import RandomSampler
from optuna.samplers._search_space import IntersectionSearchSpace
from optuna.samplers._tpe.multivariate_parzen_estimator import _MultivariateParzenEstimator
from optuna.samplers._tpe.sampler import TPESampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class CLSampler(BaseSampler):
    def __init__(
        self,
        *,
        base_sampler: BaseSampler,
        concurrency: int = 3,
        strategy: str = "min",
        param_names: List[str] = [],
    ):
        self.base_sampler = base_sampler
        self.concurrency = concurrency
        self.strategy = strategy
        self.param_names = param_names


    def sample_independent(
        self,
        study,
        trial,
        param_name,
        param_distribution,
    ):

        sampler = self.base_sampler
        if isinstance(sampler, TPESampler):
            return sampler.clsamples_independent(study, trial, param_name, param_distribution, self.concurrency, self.strategy)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        return self.base_sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:
        return self.base_sampler.sample_relative(study, trial, search_space)

