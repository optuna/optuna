import math
import numpy
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import Optional  # NOQA

from optuna import distributions
from optuna import logging
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
        self.logger = logging.get_logger(__name__)

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
            shifted_low = 0
            shifted_high = param_distribution.high - param_distribution.low
            if math.fmod(shifted_high, q) != 0:
                shifted_high = (shifted_high // q) * q
                self.logger.warning('`high` of suggest_discrete_uniform is not a multiple of `q`,'
                                    ' and it will be replaced with {}.'.format(shifted_high))
            low = shifted_low - 0.5 * q
            high = shifted_high + 0.5 * q
            s = self.rng.uniform(low, high)
            v = numpy.round(s / q) * q + param_distribution.low
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

    def __getstate__(self):
        # type: () -> Dict[Any, Any]

        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, state):
        # type: (Dict[Any, Any]) -> None

        self.__dict__.update(state)
        self.logger = logging.get_logger(__name__)
