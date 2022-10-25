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


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1 + special.erf((x - mu) / max(np.sqrt(2) * sigma, EPS)))

# def _normal_cdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
#     mu, sigma = map(np.asarray, (mu, sigma))
#     denominator = x - mu
#     numerator = np.maximum(np.sqrt(2) * sigma, EPS)
#     z = denominator / numerator
#     return 0.5 * (1 + special.erf(z))

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
            ret = stats.truncnorm.rvs(a=a, b=b, loc=self._mu, scale=self._sigma, random_state=rng)
        return ret

    def log_pdf(self, x: Any) -> float:
        p_accept = _normal_cdf(self._high, self._mu, self._sigma) - _normal_cdf(
            self._low, self._mu, self._sigma)
        return -0.5 * np.log(2 * np.pi) - np.log(self._sigma) - 0.5 * ((x - self._mu) / self._sigma) ** 2 - np.log(p_accept)
        # return stats.truncnorm.logpdf(x, a=a, b=b, loc=self._mu, scale=self._sigma)


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
        # x0 = self._align_to_step(x)
        x0 = x
        low_with_margin = self._low - self._step / 2
        high_with_margin = self._high + self._step / 2
        lb = max(x0 - self._step / 2, low_with_margin)
        ub = min(x0 + self._step / 2, high_with_margin)
        return np.log(
            (_normal_cdf(ub, self._mu, self._sigma) - _normal_cdf(lb, self._mu, self._sigma)) /
            (_normal_cdf(high_with_margin, self._mu, self._sigma) - _normal_cdf(low_with_margin, self._mu, self._sigma) + EPS)
        )


class ProductDistribution(BaseProbabilityDistribution):
    def __init__(self, distributions: List[BaseProbabilityDistribution]) -> None:
        self._distributions = distributions

    def sample(self, rng: np.random.RandomState) -> List[Any]:
        return [distribution.sample(rng) for distribution in self._distributions]

    def log_pdf(self, x: List[Any]) -> float:
        return np.sum([
            distribution.log_pdf(item) for distribution, item in zip(self._distributions, x)
        ])


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
