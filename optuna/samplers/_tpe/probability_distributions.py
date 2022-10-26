import abc
from asyncore import loop
from typing import Any, Callable
from typing import List, Dict
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


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1 + special.erf((x - mu) / np.maximum(np.sqrt(2) * sigma, EPS)))

def _sample(distribution: np.ndarray, rng: np.random.RandomState, size: Tuple[int, ...] = ()) -> np.ndarray:
    _METHOD = {
        "categorical": _categorical_sample,
        "truncnorm": _truncnorm_sample,
        "discrete_truncnorm": _discrete_truncnorm_sample,
        "product": _product_sample,
        "mixture": _mixture_sample,
    }
    assert len(distribution.dtype.names) == 1
    distribution_type = distribution.dtype.names[0]
    return _METHOD[distribution_type](distribution[distribution_type], rng, size)

def _logpdf(distribution: np.ndarray, x: np.ndarray) -> np.ndarray:
    assert distribution.shape == x.shape
    _METHOD = {
        "categorical": _categorical_logpdf,
        "truncnorm": _truncnorm_logpdf,
        "discrete_truncnorm": _discrete_truncnorm_logpdf,
        "product": _product_logpdf,
        "mixture": _mixture_logpdf,
    }
    assert len(distribution.dtype.names) == 1
    distribution_type = distribution.dtype.names[0]
    return _METHOD[distribution_type](distribution[distribution_type], x)


# categorical
def _categorical_distribution(weights: np.ndarray) -> np.ndarray:
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.rec.fromarrays([weights], dtype=[("categorical", np.float64, (weights.shape[-1],))])

def _categorical_sample(weights: np.ndarray, rng: np.random.RandomState, size: Tuple[int, ...]) -> np.ndarray:
    rnd_quantile = rng.rand(*(size + weights.shape[:-1]))
    cum_probs = np.cumsum(weights, axis=-1)
    return np.sum(cum_probs < rnd_quantile[..., None], axis=-1)

def _categorical_logpdf(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.log(np.choose(x, np.moveaxis(weights, -1, 0)))


# truncnorm
def _truncnorm_distribution(mu: np.ndarray, sigma: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    inner = np.rec.fromarrays([mu, sigma, low, high], 
                            dtype=[("mu", np.float64), 
                                    ("sigma", np.float64), 
                                    ("low", np.float64), 
                                    ("high", np.float64)])
    return np.rec.fromarrays([inner], names="truncnorm")

def _truncnorm_sample(inner: np.ndarray, rng: np.random.RandomState, size: Tuple[int, ...]) -> np.ndarray:
    return stats.truncnorm.rvs(
        a=(inner["low"] - inner["mu"]) / inner["sigma"],
        b=(inner["high"] - inner["mu"]) / inner["sigma"],
        loc=inner["mu"],
        scale=inner["sigma"],
        size=size + inner.shape,
        random_state=rng,
    )

def _truncnorm_logpdf(inner: np.ndarray, x: np.ndarray) -> np.ndarray:
    p_accept = _normal_cdf(inner["high"], inner["mu"], inner["sigma"]) - _normal_cdf(
        inner["low"], inner["mu"], inner["sigma"])
    return -0.5 * np.log(2 * np.pi) - np.log(inner["sigma"]) - 0.5 * ((x - inner["mu"]) / inner["sigma"]) ** 2 - np.log(p_accept)
    # return stats.truncnorm.logpdf(
    #     x,
    #     a=(inner["low"] - inner["mu"]) / inner["sigma"],
    #     b=(inner["high"] - inner["mu"]) / inner["sigma"],
    #     loc=inner["mu"],
    #     scale=inner["sigma"],
    # )


# discrete_truncnorm
def _discrete_truncnorm_distribution(mu: np.ndarray, sigma: np.ndarray, low: np.ndarray, high: np.ndarray, step: np.ndarray) -> np.ndarray:
    inner = np.rec.fromarrays([mu, sigma, low, high, step], 
                                dtype=[("mu", np.float64), 
                                        ("sigma", np.float64), 
                                        ("low", np.float64), 
                                        ("high", np.float64), 
                                        ("step", np.float64)])
    return np.rec.fromarrays([inner], names="discrete_truncnorm")

def _discrete_truncnorm_sample(inner: np.ndarray, rng: np.random.RandomState, size: Tuple[int, ...]) -> np.ndarray:
    samples = stats.truncnorm.rvs(
        a=(inner["low"] - inner["step"] / 2 - inner["mu"]) / inner["sigma"],
        b=(inner["high"] + inner["step"] / 2 - inner["mu"]) / inner["sigma"],
        loc=inner["mu"],
        scale=inner["sigma"],
        size=size + inner.shape,
        random_state=rng,
    )
    return np.clip(np.round(samples / inner["step"]) * inner["step"],
                    inner["low"], inner["high"])

def _discrete_truncnorm_logpdf(inner: np.ndarray, x: np.ndarray) -> np.ndarray:
    low_with_margin = inner["low"] - inner["step"] / 2
    high_with_margin = inner["high"] + inner["step"] / 2
    lb = np.maximum(x - inner["step"] / 2, low_with_margin)
    ub = np.minimum(x + inner["step"] / 2, high_with_margin)
    mu = inner["mu"]
    sigma = inner["sigma"]
    return np.log(
        (_normal_cdf(ub, mu, sigma) - _normal_cdf(lb, mu, sigma)) /
        (_normal_cdf(high_with_margin, mu, sigma) - _normal_cdf(low_with_margin, mu, sigma) + EPS)
    )

    # a = (inner["low"] - inner["step"] / 2 - inner["mu"]) / inner["sigma"]
    # b = (inner["high"] + inner["step"] / 2 - inner["mu"]) / inner["sigma"]
    # lb = x - inner["step"] / 2
    # ub = x + inner["step"] / 2
    # return np.log(stats.truncnorm.cdf(ub, a=a, b=b, loc=inner["mu"], scale=inner["sigma"]) 
    #             - stats.truncnorm.cdf(lb, a=a, b=b, loc=inner["mu"], scale=inner["sigma"]))


# product
def _product_distribution(distributions: Dict[str, np.ndarray]) -> np.ndarray:
    inner = np.rec.fromarrays(list(distributions.values()), names=list(distributions.keys()))
    return np.rec.fromarrays([inner], names="product")

def _product_sample(inner: np.ndarray, rng: np.random.RandomState, size: Tuple[int, ...]) -> np.ndarray:
    params = inner.dtype.names
    return np.rec.fromarrays([_sample(inner[param], rng, size) for param in params],
                                names=params)

def _product_logpdf(inner: np.ndarray, x: np.ndarray) -> np.ndarray:
    params = inner.dtype.names
    return np.sum([_logpdf(inner[param], x[param]) for param in params], axis=0)


# mixture
def _mixture_distribution(distributions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights /= np.sum(weights, axis=-1, keepdims=True)
    inner = np.rec.fromarrays([distributions, weights], names=["distribution", "weight"])
    return np.rec.fromarrays([inner], dtype=[("mixture", inner.dtype, (inner.shape[-1],))])

def _mixture_sample(inner: np.ndarray, rng: np.random.RandomState, size: Tuple[int, ...]) -> np.ndarray:
    active_indices = _categorical_sample(inner["weight"], rng, size)
    active_dists = np.choose(active_indices, np.moveaxis(inner["distribution"], -1, 0))
    return _sample(active_dists, rng, size=())

def _mixture_logpdf(inner: np.ndarray, x: np.ndarray) -> np.ndarray:
    return special.logsumexp(
            _logpdf(inner["distribution"], 
                    np.broadcast_to(x[..., None], inner["distribution"].shape, subok=True))
                     + np.log(inner["weight"])[(None,) * x.ndim + (slice(None),)],
                        axis=-1)

# class BaseProbabilityDistribution(metaclass=abc.ABCMeta):


#     @abc.abstractmethod
#     def sample(self, rng: np.random.RandomState) -> Any:
#         raise NotImplementedError()

#     # TODO(contramundum53): Consider vectorization with numpy.
#     @abc.abstractmethod
#     def log_pdf(self, x: Any) -> float:
#         raise NotImplementedError()


# class CategoricalDistribution(BaseProbabilityDistribution):
#     def __init__(self, weights: np.ndarray) -> None:
#         self._weights = weights / np.sum(weights)

#     def sample(self, rng: np.random.RandomState) -> int:
#         return rng.choice(len(self._weights), p=self._weights)

#     def log_pdf(self, x: int) -> float:
#         return np.log(self._weights[x])


# def _normal_cdf(x: float, mu: float, sigma: float) -> float:
#     return 0.5 * (1 + special.erf((x - mu) / max(np.sqrt(2) * sigma, EPS)))

# # def _normal_cdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
# #     mu, sigma = map(np.asarray, (mu, sigma))
# #     denominator = x - mu
# #     numerator = np.maximum(np.sqrt(2) * sigma, EPS)
# #     z = denominator / numerator
# #     return 0.5 * (1 + special.erf(z))

# class UnivariateGaussianDistribution(BaseProbabilityDistribution):
#     def __init__(self, mu: float, sigma: float, low: float, high: float) -> None:
#         self._mu = mu
#         self._sigma = max(sigma, EPS)
#         self._low = low
#         self._high = high

#     def sample(self, rng: np.random.RandomState) -> Any:
#         if self._low >= self._high:
#             return self._low
#         a = (self._low - self._mu) / self._sigma
#         b = (self._high - self._mu) / self._sigma
#         ret = float("nan")
#         while not self._low <= ret <= self._high:
#             ret = stats.truncnorm.rvs(a=a, b=b, loc=self._mu, scale=self._sigma, random_state=rng)
#         return ret

#     def log_pdf(self, x: Any) -> float:
#         p_accept = _normal_cdf(self._high, self._mu, self._sigma) - _normal_cdf(
#             self._low, self._mu, self._sigma)
#         return -0.5 * np.log(2 * np.pi) - np.log(self._sigma) - 0.5 * ((x - self._mu) / self._sigma) ** 2 - np.log(p_accept)
#         # return stats.truncnorm.logpdf(x, a=a, b=b, loc=self._mu, scale=self._sigma)


# class DiscreteUnivariateGaussianDistribution(BaseProbabilityDistribution):
#     def __init__(self, mu: float, sigma: float, low: float, high: float, step: float) -> None:
#         self._mu = mu
#         self._sigma = sigma
#         self._low = low
#         self._high = high
#         self._step = step

#     def _align_to_step(self, x: float) -> float:
#         return np.clip(np.round((x - self._low) / self._step) * self._step + self._low,
#                         self._low, self._high)

#     def sample(self, rng: np.random.RandomState) -> Any:
#         gaussian = UnivariateGaussianDistribution(self._mu, self._sigma, self._low - self._step / 2, self._high + self._step / 2)
#         return self._align_to_step(gaussian.sample(rng))

#     def log_pdf(self, x: Any) -> float:
#         # x0 = self._align_to_step(x)
#         x0 = x
#         low_with_margin = self._low - self._step / 2
#         high_with_margin = self._high + self._step / 2
#         lb = max(x0 - self._step / 2, low_with_margin)
#         ub = min(x0 + self._step / 2, high_with_margin)
#         return np.log(
#             (_normal_cdf(ub, self._mu, self._sigma) - _normal_cdf(lb, self._mu, self._sigma)) /
#             (_normal_cdf(high_with_margin, self._mu, self._sigma) - _normal_cdf(low_with_margin, self._mu, self._sigma) + EPS)
#         )


# class ProductDistribution(BaseProbabilityDistribution):
#     def __init__(self, distributions: List[BaseProbabilityDistribution]) -> None:
#         self._distributions = distributions

#     def sample(self, rng: np.random.RandomState) -> List[Any]:
#         return [distribution.sample(rng) for distribution in self._distributions]

#     def log_pdf(self, x: List[Any]) -> float:
#         return np.sum([
#             distribution.log_pdf(item) for distribution, item in zip(self._distributions, x)
#         ])


# class MixtureDistribution(BaseProbabilityDistribution):
#     def __init__(self, weights_and_distributions: List[Tuple[float, BaseProbabilityDistribution]]):
#         self._weights = np.array([weight for weight, _ in weights_and_distributions])
#         self._weights /= np.sum(self._weights)
#         self._distributions = [distribution for _, distribution in weights_and_distributions]

#     def sample(self, rng: np.random.RandomState) -> List[Any]:
#         index = rng.choice(len(self._weights), p=self._weights)
#         return self._distributions[index].sample(rng)

#     def log_pdf(self, x: Any) -> float:
#         return special.logsumexp(
#             [
#                 distribution.log_pdf(x) + np.log(weight)
#                 for weight, distribution in zip(self._weights, self._distributions)
#             ]
#         )
