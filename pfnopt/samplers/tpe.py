import math
import numpy
from typing import List  # NOQA
from typing import Optional  # NOQA

from pfnopt import distributions  # NOQA
from pfnopt.samplers import _hyperopt
from pfnopt.samplers import base
from pfnopt.samplers import random
from pfnopt.storages.base import BaseStorage  # NOQA


class TPESampler(base.BaseSampler):

    def __init__(self,
                 prior_weight=_hyperopt.default_prior_weight,
                 n_startup_trials=_hyperopt.default_n_startup_trials,
                 n_ei_candidates=_hyperopt.default_n_ei_candidates,
                 gamma=_hyperopt.default_gamma,
                 seed=None):
        # type: (float, int, int, float, Optional[int]) -> None
        self.prior_weight = prior_weight
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma
        self.seed = seed

        self.rng = numpy.random.RandomState(seed)
        self.random_sampler = random.RandomSampler(seed=seed)

    def sample(self, storage, study_id, param_name, param_distribution):
        # type: (BaseStorage, int, str, distributions.BaseDistribution) -> float
        observation_pairs = storage.get_trial_param_result_pairs(
            study_id, param_name)
        n = len(observation_pairs)

        # TODO(Akiba): this behavior is slightly different from hyperopt
        if n < self.n_startup_trials:
            return self.random_sampler.sample(storage, study_id, param_name, param_distribution)

        below_param_values, above_param_values = _hyperopt.ap_filter_trials(
            range(n), [p[0] for p in observation_pairs],
            range(n), [p[1] for p in observation_pairs],
            self.gamma)

        if isinstance(param_distribution, distributions.UniformDistribution):
            return self._sample_uniform(
                param_distribution, below_param_values, above_param_values)
        elif isinstance(param_distribution, distributions.LogUniformDistribution):
            return self._sample_loguniform(
                param_distribution, below_param_values, above_param_values)
        elif isinstance(param_distribution, distributions.QUniformDistribution):
            return self._sample_quniform(
                param_distribution, below_param_values, above_param_values)
        elif isinstance(param_distribution, distributions.CategoricalDistribution):
            return self._sample_categorical(
                param_distribution, below_param_values, above_param_values)
        else:
            raise NotImplementedError

    def _sample_uniform(self, distribution, below, above):
        # type: (distributions.UniformDistribution, List[float], List[float]) -> float
        return _hyperopt.sample_uniform(
            obs_below=below, obs_above=above, prior_weight=self.prior_weight,
            low=distribution.low, high=distribution.high,
            size=(self.n_ei_candidates,), rng=self.rng)

    def _sample_loguniform(self, distribution, below, above):
        # type: (distributions.LogUniformDistribution, List[float], List[float]) -> float

        return _hyperopt.sample_loguniform(
            obs_below=below, obs_above=above, prior_weight=self.prior_weight,
            # `sample_loguniform` generates values in [exp(low), exp(high)]
            low=math.log(distribution.low),
            high=math.log(distribution.high),
            size=(self.n_ei_candidates,), rng=self.rng)

    def _sample_quniform(self, distribution, below, above):
        # type: (distributions.QUniformDistribution, List[float], List[float]) -> float
        return _hyperopt.sample_quniform(
            obs_below=below, obs_above=above, prior_weight=self.prior_weight,
            low=distribution.low, high=distribution.high,
            size=(self.n_ei_candidates,), rng=self.rng, q=distribution.q)

    def _sample_categorical(self, distribution, below, above):
        # type: (distributions.CategoricalDistribution, List[float], List[float]) -> float
        choices = distribution.choices
        below = list(map(int, below))
        above = list(map(int, above))
        idx = _hyperopt.sample_categorical(
            obs_below=below, obs_above=above, prior_weight=self.prior_weight,
            upper=len(choices), size=(self.n_ei_candidates, ), rng=self.rng)
        return int(idx)
