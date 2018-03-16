import numpy
from typing import Optional  # NOQA

from pfnopt import distributions
from pfnopt.samplers.base import BaseSampler
from pfnopt.storages.base import BaseStorage  # NOQA


class RandomSampler(BaseSampler):

    def __init__(self, seed=None):
        # type: (Optional[int]) -> None
        self.seed = seed
        self.rng = numpy.random.RandomState(seed)

    def sample(self, storage, study_id, param_name, param_distribution):
        # type: (BaseStorage, int, str, distributions.BaseDistribution) -> float
        if isinstance(param_distribution, distributions.UniformDistribution):
            return self.rng.uniform(param_distribution.low, param_distribution.high)
        elif isinstance(param_distribution, distributions.LogUniformDistribution):
            log_low = numpy.log(param_distribution.low)
            log_high = numpy.log(param_distribution.high)
            return numpy.exp(self.rng.uniform(log_low, log_high))
        elif isinstance(param_distribution, distributions.CategoricalDistribution):
            choices = param_distribution.choices
            return self.rng.randint(len(choices))
        else:
            raise NotImplementedError
