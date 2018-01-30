import math
import numpy

from . import base_sampler


class RandomSampler(base_sampler.BaseSampler):

    def __init__(self, seed=None):
        self.seed = seed

        self.rng = numpy.random.RandomState(seed)

    def sample(self, distribution, observation_pairs):
        kind = distribution['kind']

        if kind == 'uniform':
            return self.rng.uniform(distribution['low'], distribution['high'])
        elif kind == 'loguniform':
            log_low = numpy.log(distribution['low'])
            log_high = numpy.log(distribution['high'])
            return numpy.exp(self.rng.uniform(log_low, log_high))
        else:
            raise NotImplementedError
