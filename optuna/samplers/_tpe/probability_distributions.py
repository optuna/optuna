from typing import Dict
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

# categorical
def _categorical_distribution(weights: np.ndarray) -> np.ndarray:
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.rec.fromarrays([weights], dtype=[("categorical", np.float64, (weights.shape[-1],))])

def _sample_categorical(weights: np.ndarray, rng: np.random.RandomState, size: Tuple[int, ...]) -> np.ndarray:
    rnd_quantile = rng.rand(*(size + weights.shape[:-1]))
    cum_probs = np.cumsum(weights, axis=-1)
    return np.sum(cum_probs < rnd_quantile[..., None], axis=-1)

def _logpdf_categorical(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.log(np.choose(x, np.moveaxis(weights, -1, 0)))


# truncnorm
def _truncnorm_distribution(mu: np.ndarray, sigma: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    inner = np.rec.fromarrays([mu, sigma, low, high], 
                            dtype=[("mu", np.float64), 
                                    ("sigma", np.float64), 
                                    ("low", np.float64), 
                                    ("high", np.float64)])
    return np.rec.fromarrays([inner], names="truncnorm")


EPS = 1e-12
def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1 + special.erf((x - mu) / np.maximum(np.sqrt(2) * sigma, EPS)))

def _sample_truncnorm(inner: np.ndarray, rng: np.random.RandomState, size: Tuple[int, ...]) -> np.ndarray:
    mu, sigma, low, high = inner["mu"], inner["sigma"], inner["low"], inner["high"]
    return stats.truncnorm.rvs(
        a=(low - mu) / sigma,
        b=(high - mu) / sigma,
        loc=mu,
        scale=sigma,
        size=size + inner.shape,
        random_state=rng,
    )

def _logpdf_truncnorm(inner: np.ndarray, x: np.ndarray) -> np.ndarray:
    mu, sigma, low, high = inner["mu"], inner["sigma"], inner["low"], inner["high"]
    p_accept = _normal_cdf(high, mu, sigma) - _normal_cdf(low, mu, sigma)
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2 - np.log(p_accept)


# discrete_truncnorm
def _discrete_truncnorm_distribution(mu: np.ndarray, sigma: np.ndarray, low: np.ndarray, high: np.ndarray, step: np.ndarray) -> np.ndarray:
    inner = np.rec.fromarrays([mu, sigma, low, high, step], 
                                dtype=[("mu", np.float64), 
                                        ("sigma", np.float64), 
                                        ("low", np.float64), 
                                        ("high", np.float64), 
                                        ("step", np.float64)])
    return np.rec.fromarrays([inner], names="discrete_truncnorm")

def _sample_discrete_truncnorm(inner: np.ndarray, rng: np.random.RandomState, size: Tuple[int, ...]) -> np.ndarray:
    mu, sigma, low, high, step = inner["mu"], inner["sigma"], inner["low"], inner["high"], inner["step"]
    samples = stats.truncnorm.rvs(
        a=(low - step / 2 - mu) / sigma,
        b=(high + step / 2 - mu) / sigma,
        loc=mu,
        scale=sigma,
        size=size + inner.shape,
        random_state=rng,
    )
    return np.clip(np.round(samples / step) * step, low, high)

def _logpdf_discrete_truncnorm(inner: np.ndarray, x: np.ndarray) -> np.ndarray:
    mu, sigma, low, high, step = inner["mu"], inner["sigma"], inner["low"], inner["high"], inner["step"]
    return np.log(
        (_normal_cdf(x + step / 2, mu, sigma) - _normal_cdf(x - step / 2, mu, sigma)) /
        (_normal_cdf(high + step / 2, mu, sigma) - _normal_cdf(low - step / 2, mu, sigma) + EPS)
    )


def _sample_univariate(distribution: np.ndarray, rng: np.random.RandomState, size: Tuple[int, ...] = ()) -> np.ndarray:
    _METHOD = {
        "categorical": _sample_categorical,
        "truncnorm": _sample_truncnorm,
        "discrete_truncnorm": _sample_discrete_truncnorm,
    }
    assert len(distribution.dtype.names) == 1
    distribution_type = distribution.dtype.names[0]
    return _METHOD[distribution_type](distribution[distribution_type], rng, size)

def _logpdf_univariate(distribution: np.ndarray, x: np.ndarray) -> np.ndarray:
    _METHOD = {
        "categorical": _logpdf_categorical,
        "truncnorm": _logpdf_truncnorm,
        "discrete_truncnorm": _logpdf_discrete_truncnorm,
    }
    assert len(distribution.dtype.names) == 1
    distribution_type = distribution.dtype.names[0]
    return _METHOD[distribution_type](distribution[distribution_type], x)




# product
def _product_distribution(distributions: Dict[str, np.ndarray]) -> np.ndarray:
    return np.rec.fromarrays(list(distributions.values()), names=list(distributions.keys()))

def _sample_product(product_dist: np.ndarray, rng: np.random.RandomState, size: Tuple[int, ...] = ()) -> np.ndarray:
    params = product_dist.dtype.names
    return np.rec.fromarrays([_sample_univariate(product_dist[param], rng, size) for param in params],
                                names=params)

def _logpdf_product(product_dist: np.ndarray, x: np.ndarray) -> np.ndarray:
    params = product_dist.dtype.names
    return np.sum([_logpdf_univariate(product_dist[param], x[param]) for param in params], axis=0)


# mixture
def _mixture_distribution(weights: np.ndarray, distributions: np.ndarray) -> np.ndarray:
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.rec.fromarrays([weights, distributions], names=[ "weight", "distribution"])


def _sample_mixture(mixture_dist: np.ndarray, rng: np.random.RandomState, size: Tuple[int, ...] = ()) -> np.ndarray:
    active_indices = _sample_categorical(mixture_dist["weight"], rng, size)
    active_dists = np.choose(active_indices, mixture_dist["distribution"].T)
    return _sample_product(active_dists, rng, size=())

def _logpdf_mixture(mixture_dist: np.ndarray, x: np.ndarray) -> np.ndarray:
    return special.logsumexp(
                        _logpdf_product(mixture_dist["distribution"], x[..., None])
                            + np.expand_dims(np.log(mixture_dist["weight"]), tuple(range(x.ndim))), 
                 axis=-1)
