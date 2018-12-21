import numpy
from typing import Optional  # NOQA

from optuna import distributions
from optuna.samplers.base import BaseSampler
from optuna.storages.base import BaseStorage  # NOQA


class RandomSampler(BaseSampler):
    """Sampler using random sampling.

    Example:

        .. code::

            >>> study = optuna.create_study(sampler=RandomSampler())
            >>> study.optimize(objective, direction='minimize')

        Args:
            seed: Seed for random number generator.
    """

    def __init__(self, seed=None):
        # type: (Optional[int]) -> None

        self.seed = seed
        self.rng = numpy.random.RandomState(seed)

    def sample(self, storage, study_id, param_name, param_distribution):
        # type: (BaseStorage, int, str, distributions.BaseDistribution) -> float
        """Please consult the documentation for :func:`BaseSampler.sample`."""

        if isinstance(param_distribution, distributions.UniformDistribution):
            return self.rng.uniform(param_distribution.low, param_distribution.high)
        elif isinstance(param_distribution, distributions.LogUniformDistribution):
            log_low = numpy.log(param_distribution.low)
            log_high = numpy.log(param_distribution.high)
            return numpy.exp(self.rng.uniform(log_low, log_high))
        elif isinstance(param_distribution, distributions.DiscreteUniformDistribution):
            q = param_distribution.q
            low = param_distribution.low - 0.5 * q
            high = param_distribution.high + 0.5 * q
            s = self.rng.uniform(low, high)
            v = numpy.round(s / q) * q
            # v may slightly exceed range due to round-off errors.
            return min(max(v, param_distribution.low), param_distribution.high)
        elif isinstance(param_distribution, distributions.IntUniformDistribution):
            # numpy.random.randint includes low but excludes high.
            return self.rng.randint(param_distribution.low, param_distribution.high + 1)
        elif isinstance(param_distribution, distributions.CategoricalDistribution):
            choices = param_distribution.choices
            return self.rng.randint(len(choices))
        else:
            raise NotImplementedError
