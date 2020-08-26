from typing import Any
from typing import Dict
from typing import Optional

import numpy

from optuna import distributions
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.trial import FrozenTrial


class RandomSampler(BaseSampler):
    """Sampler using random sampling.

    This sampler is based on *independent sampling*.
    See also :class:`~optuna.samplers.BaseSampler` for more details of 'independent sampling'.

    Example:

        .. testcode::

            import optuna
            from optuna.samplers import RandomSampler

            def objective(trial):
                x = trial.suggest_uniform('x', -5, 5)
                return x**2

            study = optuna.create_study(sampler=RandomSampler())
            study.optimize(objective, n_trials=10)

        Args:
            seed: Seed for random number generator.
    """

    def __init__(self, seed: Optional[int] = None) -> None:

        self._rng = numpy.random.RandomState(seed)

    def reseed_rng(self) -> None:

        self._rng = numpy.random.RandomState()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:

        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:

        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:

        if isinstance(param_distribution, distributions.UniformDistribution):
            return self._rng.uniform(param_distribution.low, param_distribution.high)
        elif isinstance(param_distribution, distributions.LogUniformDistribution):
            log_low = numpy.log(param_distribution.low)
            log_high = numpy.log(param_distribution.high)
            return float(numpy.exp(self._rng.uniform(log_low, log_high)))
        elif isinstance(param_distribution, distributions.DiscreteUniformDistribution):
            q = param_distribution.q
            r = param_distribution.high - param_distribution.low
            # [low, high] is shifted to [0, r] to align sampled values at regular intervals.
            low = 0 - 0.5 * q
            high = r + 0.5 * q
            s = self._rng.uniform(low, high)
            v = numpy.round(s / q) * q + param_distribution.low
            # v may slightly exceed range due to round-off errors.
            return float(min(max(v, param_distribution.low), param_distribution.high))
        elif isinstance(param_distribution, distributions.IntUniformDistribution):
            # [low, high] is shifted to [0, r] to align sampled values at regular intervals.
            r = (param_distribution.high - param_distribution.low) / param_distribution.step
            # numpy.random.randint includes low but excludes high.
            s = self._rng.randint(0, r + 1)
            v = s * param_distribution.step + param_distribution.low
            return int(v)
        elif isinstance(param_distribution, distributions.IntLogUniformDistribution):
            log_low = numpy.log(param_distribution.low - 0.5)
            log_high = numpy.log(param_distribution.high + 0.5)
            s = numpy.exp(self._rng.uniform(log_low, log_high))
            v = numpy.round(s)
            return int(min(max(v, param_distribution.low), param_distribution.high))
        elif isinstance(param_distribution, distributions.CategoricalDistribution):
            choices = param_distribution.choices
            index = self._rng.randint(0, len(choices))
            return choices[index]
        else:
            raise NotImplementedError
