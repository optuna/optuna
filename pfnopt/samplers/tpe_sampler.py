import numpy

from pfnopt.samplers import _hyperopt
from pfnopt.samplers import base_sampler
from pfnopt.samplers import random_sampler


class TPESampler(base_sampler.BaseSampler):

    def __init__(self,
                 prior_weight=_hyperopt.default_prior_weight,
                 n_startup_trials=_hyperopt.default_n_startup_trials,
                 n_ei_candidates=_hyperopt.default_n_ei_candidates,
                 gamma=_hyperopt.default_gamma,
                 seed=None):
        self.prior_weight = prior_weight
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma
        self.seed = seed

        self.rng = numpy.random.RandomState(seed)
        self.random_sampler = random_sampler.RandomSampler(seed=seed)

    def sample(self, distribution, observation_pairs):
        n = len(observation_pairs)

        # TODO: this behavior is slightly different from hyperopt
        if n < self.n_startup_trials:
            return self.random_sampler.sample(distribution, observation_pairs)

        below_param_values, above_param_values = _hyperopt.ap_filter_trials(
            range(n), [p[0] for p in observation_pairs],
            range(n), [p[1] for p in observation_pairs],
            self.gamma)

        if distribution['kind'] == 'uniform':
            return self._sample_uniform(distribution, below_param_values, above_param_values)
        else:
            raise NotImplementedError

    def _sample_uniform(self, distribution, below, above):
        return _hyperopt.iwiwi_uniform_sampler(
            obs_below=below, obs_above=above, prior_weight=self.prior_weight,
            low=distribution['low'], high=distribution['high'],
            size=(self.n_ei_candidates,), rng=self.rng)
