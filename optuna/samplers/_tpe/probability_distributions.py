from typing import List
from typing import NamedTuple
from typing import TYPE_CHECKING
from typing import Union

import numpy as np

from optuna._imports import _LazyImport


if TYPE_CHECKING:
    import scipy.special as special
    import scipy.stats as stats
else:
    special = _LazyImport("scipy.special")
    stats = _LazyImport("scipy.stats")


class _BatchedCategoricalDistributions(NamedTuple):
    weights: np.ndarray


class _BatchedTruncNormDistributions(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray
    low: float  # Currently, low and high do not change per trial.
    high: float


class _BatchedDiscreteTruncNormDistributions(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray
    low: float  # Currently, low, high and step do not change per trial.
    high: float
    step: float


_BatchedDistributionUnion = Union[
    _BatchedCategoricalDistributions,
    _BatchedTruncNormDistributions,
    _BatchedDiscreteTruncNormDistributions,
]

EPS = 1e-12


def _normal_cdf(
    x: Union[float, np.ndarray], mu: Union[float, np.ndarray], sigma: Union[float, np.ndarray]
) -> np.ndarray:
    return 0.5 * (1 + special.erf((x - mu) / np.maximum(np.sqrt(2) * sigma, EPS)))


class _MixtureOfProductDistribution(NamedTuple):
    weights: np.ndarray
    distributions: List[_BatchedDistributionUnion]

    def sample(self, rng: np.random.RandomState, batch_size: int) -> np.ndarray:
        active_indices = rng.choice(len(self.weights), p=self.weights, size=batch_size)

        ret = np.empty((batch_size, len(self.distributions)), dtype=np.float64)
        for i, d in enumerate(self.distributions):
            if isinstance(d, _BatchedCategoricalDistributions):
                active_weights = d.weights[active_indices, :]
                rnd_quantile = rng.rand(batch_size)
                cum_probs = np.cumsum(active_weights, axis=-1)
                assert np.isclose(cum_probs[:, -1], 1).all()
                cum_probs[:, -1] = 1  # Avoid numerical errors.
                ret[:, i] = np.sum(cum_probs < rnd_quantile[:, None], axis=-1)
            elif isinstance(d, _BatchedTruncNormDistributions):
                active_mus = d.mu[active_indices]
                active_sigmas = d.sigma[active_indices]
                ret[:, i] = stats.truncnorm.rvs(
                    a=(d.low - active_mus) / active_sigmas,
                    b=(d.high - active_mus) / active_sigmas,
                    loc=active_mus,
                    scale=active_sigmas,
                    random_state=rng,
                )
            elif isinstance(d, _BatchedDiscreteTruncNormDistributions):
                active_mus = d.mu[active_indices]
                active_sigmas = d.sigma[active_indices]
                samples = stats.truncnorm.rvs(
                    a=(d.low - d.step / 2 - active_mus) / active_sigmas,
                    b=(d.high + d.step / 2 - active_mus) / active_sigmas,
                    loc=active_mus,
                    scale=active_sigmas,
                    random_state=rng,
                )
                ret[:, i] = np.clip(
                    d.low + np.round((samples - d.low) / d.step) * d.step, d.low, d.high
                )
            else:
                assert False

        return ret

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        batch_size, n_vars = x.shape
        log_pdfs = np.empty((batch_size, len(self.weights), n_vars), dtype=np.float64)
        for i, d in enumerate(self.distributions):
            xi = x[:, i]
            if isinstance(d, _BatchedCategoricalDistributions):
                log_pdfs[:, :, i] = np.log(
                    np.take_along_axis(
                        d.weights[None, :, :], xi[:, None, None].astype(np.int64), axis=-1
                    )
                )[:, :, 0]
            elif isinstance(d, _BatchedTruncNormDistributions):
                p_accept: np.ndarray = _normal_cdf(d.high, d.mu, d.sigma) - _normal_cdf(
                    d.low, d.mu, d.sigma
                )
                log_pdfs[:, :, i] = (
                    -0.5 * np.log(2 * np.pi)
                    - np.log(d.sigma[None, :])
                    - 0.5 * ((xi[:, None] - d.mu[None, :]) / d.sigma[None, :]) ** 2
                    - np.log(p_accept[None, :])
                )
            elif isinstance(d, _BatchedDiscreteTruncNormDistributions):
                log_pdfs[:, :, i] = np.log(
                    (
                        _normal_cdf(xi[:, None] + d.step / 2, d.mu[None, :], d.sigma[None, :])
                        - _normal_cdf(xi[:, None] - d.step / 2, d.mu[None, :], d.sigma[None, :])
                        + EPS
                    )
                    / (
                        _normal_cdf(d.high + d.step / 2, d.mu[None, :], d.sigma[None, :])
                        - _normal_cdf(d.low - d.step / 2, d.mu[None, :], d.sigma[None, :])
                        + EPS
                    )
                )
            else:
                assert False

        return special.logsumexp(
            np.sum(log_pdfs, axis=-1) + np.log(self.weights[None, :]), axis=-1
        )
