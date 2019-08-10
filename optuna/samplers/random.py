import numpy

from optuna import distributions
from optuna.samplers.base import BaseSampler
from optuna import types

if types.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import InTrialStudy  # NOQA


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

    def infer_relative_search_space(self, study, trial):
        # type: (InTrialStudy, FrozenTrial) -> Dict[str, BaseDistribution]

        return {}

    def sample_relative(self, study, trial, search_space):
        # type: (InTrialStudy, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, Any]

        return {}

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (InTrialStudy, FrozenTrial, str, distributions.BaseDistribution) -> Any
        """Please consult the documentation for :func:`BaseSampler.sample_independent`."""

        if isinstance(param_distribution, distributions.UniformDistribution):
            return self.rng.uniform(param_distribution.low, param_distribution.high)
        elif isinstance(param_distribution, distributions.LogUniformDistribution):
            log_low = numpy.log(param_distribution.low)
            log_high = numpy.log(param_distribution.high)
            return float(numpy.exp(self.rng.uniform(log_low, log_high)))
        elif isinstance(param_distribution, distributions.DiscreteUniformDistribution):
            q = param_distribution.q
            r = param_distribution.high - param_distribution.low
            # [low, high] is shifted to [0, r] to align sampled values at regular intervals.
            low = 0 - 0.5 * q
            high = r + 0.5 * q
            s = self.rng.uniform(low, high)
            v = numpy.round(s / q) * q + param_distribution.low
            # v may slightly exceed range due to round-off errors.
            return float(min(max(v, param_distribution.low), param_distribution.high))
        elif isinstance(param_distribution, distributions.IntUniformDistribution):
            # numpy.random.randint includes low but excludes high.
            return self.rng.randint(param_distribution.low, param_distribution.high + 1)
        elif isinstance(param_distribution, distributions.CategoricalDistribution):
            choices = param_distribution.choices
            index = self.rng.randint(0, len(choices))
            return choices[index]
        else:
            raise NotImplementedError
