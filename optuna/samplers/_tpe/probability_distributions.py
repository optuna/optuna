import abc
from asyncore import loop
from typing import Any
from typing import List
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np

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
        if self._low >= self._high:
            return self._low
        a = (self._low - self._mu) / self._sigma
        b = (self._high - self._mu) / self._sigma
        ret = float("nan")
        while not self._low <= ret <= self._high:
            ret = stats.truncnorm(a, b, loc=self._mu, scale=self._sigma).rvs(random_state=rng)
        return ret

    def log_pdf(self, x: Any) -> float:
        a = (self._low - self._mu) / self._sigma
        b = (self._high - self._mu) / self._sigma
        return stats.truncnorm(a, b, loc=self._mu, scale=self._sigma).logpdf(x)


class DiscreteUnivariateGaussianDistribution(BaseProbabilityDistribution):
    def __init__(self, mu: float, sigma: float, low: float, high: float, step: float) -> None:
        self._mu = mu
        self._sigma = sigma
        self._low = low
        self._high = high
        self._step = step

    def _align_to_step(self, x: float) -> float:
        return np.clip(np.round((x - self._low) / self._step) * self._step + self._low,
                        self._low, self._high)

    def sample(self, rng: np.random.RandomState) -> Any:
        gaussian = UnivariateGaussianDistribution(self._mu, self._sigma, self._low - self._step / 2, self._high + self._step / 2)
        return self._align_to_step(gaussian.sample(rng))

    def log_pdf(self, x: Any) -> float:
        a = (self._low - self._step / 2 - self._mu) / self._sigma
        b = (self._high + self._step / 2 - self._mu) / self._sigma
        x0 = self._align_to_step(x)
        lb = (x0 - self._step / 2 - self._mu) / self._sigma
        ub = (x0 + self._step / 2 - self._mu) / self._sigma
        return np.log(stats.truncnorm(a, b).cdf(ub) - stats.truncnorm(a, b).cdf(lb))


class ProductDistribution(BaseProbabilityDistribution):
    def __init__(self, distributions: List[BaseProbabilityDistribution]) -> None:
        self._distributions = distributions

    def sample(self, rng: np.random.RandomState) -> List[Any]:
        return [distribution.sample(rng) for distribution in self._distributions]

    def log_pdf(self, x: List[Any]) -> float:
        return sum(
            distribution.log_pdf(item) for distribution, item in zip(self._distributions, x)
        )


class MixtureDistribution(BaseProbabilityDistribution):
    def __init__(self, weights_and_distributions: List[Tuple[float, BaseProbabilityDistribution]]):
        self._weights = np.array([weight for weight, _ in weights_and_distributions])
        self._weights /= np.sum(self._weights)
        self._distributions = [distribution for _, distribution in weights_and_distributions]

    def sample(self, rng: np.random.RandomState) -> List[Any]:
        index = rng.choice(len(self._weights), p=self._weights)
        return self._distributions[index].sample(rng)

    def log_pdf(self, x: Any) -> float:
        return special.logsumexp(
            [
                distribution.log_pdf(x) + np.log(weight)
                for weight, distribution in zip(self._weights, self._distributions)
            ]
        )
