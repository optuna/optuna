# TODO: fix comment style
# TODO: organize the sode of sobol_seq
# Especially, we should re-consider how i4_sobol handles its seed using global variable.

from collections import OrderedDict
from math import exp
from math import log
from typing import Any
from typing import Dict
from typing import Optional
import warnings

import numpy as np

import optuna
from optuna import distributions
from optuna.distributions import BaseDistribution
from optuna.samplers import RandomSampler
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

import sobol_seq

_NUMERICAL_DISTRIBUTIONS = (
    distributions.UniformDistribution,
    distributions.LogUniformDistribution,
    distributions.DiscreteUniformDistribution,
    distributions.IntUniformDistribution,
    distributions.IntLogUniformDistribution,
)

_SUGGESTED_STATES = (TrialState.COMPLETE, TrialState.PRUNED)


class SobolSampler(BaseSampler):

    def __init__(
        self,
        seed: Optional[int] = None,
        search_space: Dict[str, BaseDistribution] = None,
        *,
        warn_independent_sampling: bool = True
    ) -> None:

        # handle dimentions
        # handle sample size that is not power of 2
        # probably we do not have to
        self._seed = seed
        self._random_sampler = RandomSampler(seed=seed)
        # If search_space is not specified,
        # the first trial is sampled by independent sampler.
        self._initial_search_space = search_space
        self._warn_independent_sampling = warn_independent_sampling
        self._count = 0
        self._random_shift = None

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:

        if self._initial_search_space is not None:
            return self._initial_search_space
        # If no trial was made, use sample independent.
        past_trials = study._storage.get_all_trials(study._study_id, deepcopy=False)
        past_trials = [t for t in past_trials if t.state in _SUGGESTED_STATES]
        if len(past_trials) == 0:
            return {}
        # If an initial trial was already made,
        # pick up all compatible distributions from the trial.
        else:
            return self._infer_initial_search_space(past_trials[0])

    def _infer_initial_search_space(
        self, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:

        search_space = OrderedDict()  # type: OrderedDict[str, BaseDistribution]
        for name, distribution in trial.distributions.items():
            if not isinstance(distribution, _NUMERICAL_DISTRIBUTIONS):
                if self._warn_independent_sampling:
                    warnings.warn(
                        "The distribution {} is not supported by SobolSampler."
                        "The parameter distribution should be one of the {}".format(
                            distribution, [d.__name__ for d in _NUMERICAL_DISTRIBUTIONS]
                        )
                    )
                continue
            search_space[name] = distribution

        n_params = len(search_space)
        rng = np.random.RandomState(self._seed)

        self._random_shift = rng.rand(n_params)
        self._n_initial_params = n_params
        self._initial_search_space = search_space

        return search_space

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:

        return self._random_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:

        if search_space == {}:
            return {}

        assert self._initial_search_space is not None

        samples, _ = sobol_seq.i4_sobol(self._n_initial_params, seed=1 + self._count)
        samples = self._randomize(samples)
        samples = self._transform_from_uniform(samples)
        self._count += 1
        return samples

    def _transform_from_uniform(
        self, multivariate_samples: np.ndarray
    ) -> Dict[str, np.ndarray]:

        transformed = {}
        assert self._initial_search_space is not None
        for param, uniform_sample in zip(
            self._initial_search_space.items(), multivariate_samples
        ):

            param_name, distribution = param
            assert isinstance(distribution, _NUMERICAL_DISTRIBUTIONS)

            if isinstance(distribution, distributions.UniformDistribution):
                high = distribution.high
                low = distribution.low
                sample = uniform_sample * (high - low) + low
            elif isinstance(distribution, distributions.LogUniformDistribution):
                log_low = log(distribution.low)
                log_high = log(distribution.high)
                sample = exp(uniform_sample * (log_high - log_low))
            elif isinstance(distribution, distributions.DiscreteUniformDistribution):
                q = distribution.q
                r = distribution.high - distribution.low
                # [low, high] is shifted to [0, r] to align sampled values at regular intervals.
                low = 0 - 0.5 * q
                high = r + 0.5 * q
                sample = round(uniform_sample * (high - low) / q) * q + distribution.low
            elif isinstance(distribution, distributions.IntUniformDistribution):
                q = distribution.step
                r = distribution.high - distribution.low
                # [low, high] is shifted to [0, r] to align sampled values at regular intervals.
                low = 0 - 0.5 * q
                high = r + 0.5 * q
                sample = round(uniform_sample * (high - low) / q) * q + distribution.low
            elif isinstance(distribution, distributions.IntLogUniformDistribution):
                log_low = log(distribution.low - 0.5)
                log_high = log(distribution.high + 0.5)
                sample = round(exp(sample * (log_high - log_low)))

            sample = self._clip(sample, distribution.high, distribution.low)
            transformed[param_name] = distribution.to_external_repr(sample)

        return transformed

    def _randomize(self, samples: np.ndarray) -> np.ndarray:

        samples += self._random_shift
        samples %= 1
        return samples
        
    def reseed_rng(self) -> None:
        pass

    @staticmethod
    def _clip(x: float, high: float, low: float) -> float:
        assert high >= low
        return min(high, max(low, x))


def main():

    sampler = SobolSampler()

    def obj1(t):
        r = 0
        for i in range(5):
            r += t.suggest_uniform("x{}".format(i), 1.0, 100.0)
        return r

    # for checking warning on incompatible distributions
    # obj2 = lambda t: sum([t.suggest_categorical("x{}".format(i), [1,2,3,4]) for i in range(5)])

    study = optuna.create_study(sampler=sampler)
    study.optimize(obj1, n_trials=100)


if __name__ == "__main__":
    main()
