import numpy as np

from optuna import distributions  # NOQA
from optuna.distributions import BaseDistribution  # NOQA
from optuna.samplers import base  # NOQA
from optuna.samplers import random  # NOQA
from optuna.samplers import TPESampler
from optuna.storages.base import BaseStorage  # NOQA
from optuna.structs import StudyDirection  # NOQA
from optuna import types

if types.TYPE_CHECKING:
    from typing import Optional  # NOQA


DEFAULT_START_TEMPERATURE = 1000.0
DEFAULT_TEMPERATURE_COEFFICIENT = 0.95


class SASampler(TPESampler):
    def __init__(
            self,
            consider_prior=True,  # type: bool
            prior_weight=1.0,  # type: Optional[float]
            consider_magic_clip=True,  # type: bool
            consider_endpoints=False,  # type: bool
            n_startup_trials=10,  # type: int
            n_ei_candidates=24,  # type: int
            seed=None,  # type: Optional[int]
            start_temperature=DEFAULT_START_TEMPERATURE,  # type: Optional[float]
            # type: Optional[float]
            temperature_reduction_coefficient=DEFAULT_TEMPERATURE_COEFFICIENT
    ):
        super(SASampler, self).__init__(
            consider_prior=consider_prior,
            prior_weight=prior_weight,
            consider_magic_clip=consider_magic_clip,
            consider_endpoints=consider_endpoints,
            n_startup_trials=n_startup_trials,
            n_ei_candidates=n_ei_candidates,
            seed=seed)

        self.temperature = start_temperature
        self.temperature_reduction_coefficient = temperature_reduction_coefficient

    def _regen_simulation_pairs(self, observation_pairs):
        X = []
        Y = []
        temperature = self.temperature

        for i, (p, v) in enumerate(observation_pairs):
            # Append the actual param value to Y
            Y.append(p)

            # Regenenerate X based on random trial (may not be the exact X, but comes really close)
            if not X:
                X.append(p)
            else:
                prob = np.exp((observation_pairs[i - 1][1] - v) / temperature)
                X.append(p if self.rng.uniform() <= prob else observation_pairs[i - 1][0])

            temperature *= self.temperature_reduction_coefficient

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        return X, Y

    def sample(self, storage, study_id, param_name, param_distribution):
        # type: (BaseStorage, int, str, BaseDistribution) -> float

        observation_pairs = storage.get_trial_param_result_pairs(study_id, param_name)
        if storage.get_study_direction(study_id) == StudyDirection.MAXIMIZE:
            observation_pairs = [(p, -v) for p, v in observation_pairs]

        n = len(observation_pairs)

        if n < self.n_startup_trials:
            return self.random_sampler.sample(storage, study_id, param_name, param_distribution)

        X, Y = self._regen_simulation_pairs(observation_pairs)

        if isinstance(param_distribution, distributions.UniformDistribution):
            return self._sample_uniform(param_distribution, X, Y)
        elif isinstance(param_distribution, distributions.LogUniformDistribution):
            return self._sample_loguniform(param_distribution, X, Y)
        elif isinstance(param_distribution, distributions.DiscreteUniformDistribution):
            return self._sample_discrete_uniform(param_distribution, X, Y)
        elif isinstance(param_distribution, distributions.IntUniformDistribution):
            return self._sample_int(param_distribution, X, Y)
        elif isinstance(param_distribution, distributions.CategoricalDistribution):
            return self._sample_categorical(param_distribution, X, Y)
        else:
            distribution_list = [
                distributions.UniformDistribution.__name__,
                distributions.LogUniformDistribution.__name__,
                distributions.DiscreteUniformDistribution.__name__,
                distributions.IntUniformDistribution.__name__,
                distributions.CategoricalDistribution.__name__
            ]
            raise NotImplementedError("The distribution {} is not implemented. "
                                      "The parameter distribution should be one of the {}".format(
                                          param_distribution, distribution_list))
