import numpy

from pfnopt.samplers import base_sampler


class RandomSampler(base_sampler.BaseSampler):

    def __init__(self, seed=None):
        self.seed = seed

        self.rng = numpy.random.RandomState(seed)

    def sample(self, distribution, observation_pairs):
        kind = distribution['kind']

        if kind == 'uniform':
            return self.rng.uniform(distribution['low'], distribution['high'])
        else:
            raise NotImplementedError
