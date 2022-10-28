from typing import Dict, List
from typing import Tuple
from typing import TYPE_CHECKING

import numpy as np

from optuna._imports import _LazyImport
import abc
from typing import TypeVar


if TYPE_CHECKING:
    import scipy.special as special
    import scipy.stats as stats
else:
    special = _LazyImport("scipy.special")
    stats = _LazyImport("scipy.stats")

T = TypeVar("T")
class _BaseVectorizedDistributions(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, rng: np.random.RandomState) -> np.ndarray:
        raise NotImplementedError
    
    @abc.abstractmethod
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abc.abstractmethod
    def take(self: T, indices: np.ndarray) -> T:
        raise NotImplementedError

class _VectorizedCategoricalDistributions(_BaseVectorizedDistributions):
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = weights / np.sum(weights, axis=-1, keepdims=True)

    def sample(self, rng: np.random.RandomState) -> np.ndarray:
        rnd_quantile = rng.rand(self.weights.shape[0])
        cum_probs = np.cumsum(self.weights, axis=-1)
        return np.sum(cum_probs < rnd_quantile[..., None], axis=-1)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        return np.log(np.take_along_axis(self.weights[None, :], x[..., None].astype(np.int64), axis=-1))[..., 0]

    def take(self, indices: np.ndarray) -> "_VectorizedCategoricalDistributions":
        return _VectorizedCategoricalDistributions(self.weights[indices])

EPS = 1e-12
def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1 + special.erf((x - mu) / np.maximum(np.sqrt(2) * sigma, EPS)))
class _VectorizedTruncNormDistributions(_BaseVectorizedDistributions):
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, low: np.ndarray, high: np.ndarray) -> None:
        self.mu, self.sigma, self.low, self.high = np.broadcast_arrays(mu, sigma, low, high, subok=True)

    def sample(self, rng: np.random.RandomState) -> np.ndarray:
        return stats.truncnorm.rvs(
            a=(self.low - self.mu) / self.sigma,
            b=(self.high - self.mu) / self.sigma,
            loc=self.mu,
            scale=self.sigma,
            random_state=rng,
        )
    
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        p_accept = _normal_cdf(self.high, self.mu, self.sigma) - _normal_cdf(self.low, self.mu, self.sigma)
        return -0.5 * np.log(2 * np.pi) - np.log(self.sigma) - 0.5 * ((x - self.mu) / self.sigma) ** 2 - np.log(p_accept)

    def take(self, indices: np.ndarray) -> "_VectorizedTruncNormDistributions":
        return _VectorizedTruncNormDistributions(
            self.mu[indices], self.sigma[indices], self.low[indices], self.high[indices]
        )

class _VectorizedDiscreteTruncNormDistributions(_BaseVectorizedDistributions):
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, low: np.ndarray, high: np.ndarray, step: np.ndarray) -> None:
        self.mu, self.sigma, self.low, self.high, self.step = np.broadcast_arrays(mu, sigma, low, high, step, subok=True)
    
    def sample(self, rng: np.random.RandomState) -> np.ndarray:
        samples = stats.truncnorm.rvs(
            a=(self.low - self.step / 2 - self.mu) / self.sigma,
            b=(self.high + self.step / 2 - self.mu) / self.sigma,
            loc=self.mu,
            scale=self.sigma,
            random_state=rng,
        )
        return np.clip(np.round(samples / self.step) * self.step, self.low, self.high)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        return np.log(
            (_normal_cdf(x + self.step / 2, self.mu, self.sigma) - _normal_cdf(x - self.step / 2, self.mu, self.sigma)) /
            (_normal_cdf(self.high + self.step / 2, self.mu, self.sigma) - _normal_cdf(self.low - self.step / 2, self.mu, self.sigma) + EPS)
        )

    def take(self, indices: np.ndarray) -> "_VectorizedDiscreteTruncNormDistributions":
        return _VectorizedDiscreteTruncNormDistributions(
            self.mu[indices], self.sigma[indices], self.low[indices], self.high[indices], self.step[indices]
        )

class _MixtureOfProductDistribution:
    def __init__(self,
                 weights: np.ndarray,
                 distributions: List[_BaseVectorizedDistributions]) -> None:
        self.distributions = distributions
        self.weights = weights / np.sum(weights)

    def sample(self, rng: np.random.RandomState, batch_size: int) -> np.ndarray:
        active_indices = _VectorizedCategoricalDistributions(
            np.broadcast_to(self.weights, (batch_size, self.weights.size))).sample(rng)
        
        active_dists = [d.take(active_indices) for d in self.distributions]
        return np.array([
            dist.sample(rng) for dist in active_dists]).T

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        
        return special.logsumexp(
                np.sum([dist.log_pdf(x[:, None, k]) for k, dist in enumerate(self.distributions)], axis=0)
                    + np.log(self.weights[None, :]), 
                axis=-1)
