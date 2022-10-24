import abc
from lib2to3.pytree import Base
from typing import Callable
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Any
from typing import List
from typing import TYPE_CHECKING

import numpy as np

from optuna import distributions
from optuna._imports import _LazyImport


if TYPE_CHECKING:
    import scipy.special as special
    import scipy.stats as stats
else:
    special = _LazyImport("scipy.special")
    stats = _LazyImport("scipy.stats")

EPS = 1e-12

class BaseProbabilityDistribution(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, rng: np.random.RandomState) -> Any:
        raise NotImplementedError()

    # TODO(contramundum53): Consider vectorization with numpy.
    @abc.abstractmethod
    def log_pdf(self, x: Any) -> float:
        raise NotImplementedError()

class CategoricalDistribution(BaseProbabilityDistribution):
    def __init__(self, weights: np.ndarray) -> None:
        self._weights = weights / np.sum(weights)

    def sample(self, rng: np.random.RandomState) -> int:
        return rng.choice(len(self._weights), p=self._weights)

    def log_pdf(self, x: int) -> float:
        return np.log(self._weights[x])
    
class UnivariateGaussianDistribution(BaseProbabilityDistribution):
    def __init__(self, mu: float, sigma: float, low: float, high: float) -> None:
        self._mu = mu
        self._sigma = max(sigma, EPS)
        self._low = low
        self._high = high

    def sample(self, rng: np.random.RandomState) -> Any:
        a = (self._low - self._mu) / self._sigma
        b = (self._high - self._mu) / self._sigma
        return stats.truncnorm(a, b, loc=self._mu, scale=self._sigma).rvs(random_state=rng)
    
    def log_pdf(self, x: Any) -> float:
        a = (self._low - self._mu) / self._sigma
        b = (self._high - self._mu) / self._sigma
        return stats.truncnorm(a, b, loc=self._mu, scale=self._sigma).cdf(x)

class DiscreteUnivariateGaussianDistribution(BaseProbabilityDistribution):
    def __init__(self, mu: float, sigma: float, low: float, high: float, step: float) -> None:
        self._gaussian = UnivariateGaussianDistribution(mu, sigma, low - step/2, high + step/2)
        self._step = step
    
    def _align_to_step(self, x: float) -> float:
        return np.round((x - self._gaussian._low) / self._step) * self._step + self._gaussian._low

    def sample(self, rng: np.random.RandomState) -> Any:
        return self._align_to_step(self._gaussian.sample(rng))

    def log_pdf(self, x: Any) -> float:
        a = (self._gaussian._low - self._gaussian._mu) / self._gaussian._sigma
        b = (self._gaussian._high - self._gaussian._mu) / self._gaussian._sigma
        x0 = self._align_to_step(x)
        l = (x0 - self._gaussian._low) / self._gaussian._sigma
        h = (x0 + self._gaussian._low) / self._gaussian._sigma
        return stats.truncnorm(a, b).cdf(h) - stats.truncnorm(a, b).cdf(l)


class ProductDistribution(BaseProbabilityDistribution):
    def __init__(self, distributions: List[BaseProbabilityDistribution]) -> None:
        self._distributions = distributions
    
    def sample(self, rng: np.random.RandomState) -> List[Any]:
        return [distribution.sample(rng) for distribution in self._distributions]

    def log_pdf(self, x: List[Any]) -> float:
        return sum(distribution.log_pdf(item) for distribution, item in zip(self._distributions, x))

class MixtureDistribution(BaseProbabilityDistribution):
    def __init__(self, weights_and_distributions: List[Tuple[float, BaseProbabilityDistribution]]):
        self._weights = np.array([weight for weight, _ in weights_and_distributions])
        self._weights /= np.sum(self._weights)
        self._distributions = [distribution for _, distribution in weights_and_distributions]
    
    def sample(self, rng: np.random.RandomState) -> List[Any]:
        index = rng.choice(len(self._weights), p=self._weights)
        return self._distributions[index].sample(rng)

    def log_pdf(self, x: Any) -> float:
        return special.logsumexp([distribution.log_pdf(x) + np.log(weight) for weight, distribution in zip(self._weights, self._distributions)])

