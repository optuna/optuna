from __future__ import annotations

from typing import NamedTuple
from typing import Union

import numpy as np

from optuna.samplers._tpe import _truncnorm


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


_BatchedDistributions = Union[
    _BatchedCategoricalDistributions,
    _BatchedTruncNormDistributions,
    _BatchedDiscreteTruncNormDistributions,
]


def _unique_inverse_2d(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function is a quicker version of:
        np.unique(np.concatenate([a[:, None], b[:, None]], axis=-1), return_inverse=True).
    """
    assert a.shape == b.shape and len(a.shape) == 1
    order = np.argsort(b)
    # Stable sorting is required for the tie breaking.
    order = order[np.argsort(a[order], kind="stable")]
    a_order = a[order]
    b_order = b[order]
    is_first_occurrence = np.empty_like(a, dtype=bool)
    is_first_occurrence[0] = True
    is_first_occurrence[1:] = (a_order[1:] != a_order[:-1]) | (b_order[1:] != b_order[:-1])
    inv = np.empty(a_order.size, dtype=int)
    inv[order] = np.cumsum(is_first_occurrence) - 1
    return a_order[is_first_occurrence], b_order[is_first_occurrence], inv


def _log_gauss_mass_unique(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This function reduces the log Gaussian probability mass computation by avoiding the
    duplicated evaluations using the np.unique_inverse(...) equivalent operation.
    """
    a_uniq, b_uniq, inv = _unique_inverse_2d(a.ravel(), b.ravel())
    return _truncnorm._log_gauss_mass(a_uniq, b_uniq)[inv].reshape(a.shape)


class _MixtureOfProductDistribution(NamedTuple):
    weights: np.ndarray
    distributions: list[_BatchedDistributions]

    def sample(self, rng: np.random.RandomState, batch_size: int) -> np.ndarray:
        active_indices = rng.choice(len(self.weights), p=self.weights, size=batch_size)
        ret = np.empty((batch_size, len(self.distributions)), dtype=float)
        disc_inds, numerical_inds = [], []
        numerical_dists: list[
            _BatchedTruncNormDistributions | _BatchedDiscreteTruncNormDistributions
        ] = []
        for i, d in enumerate(self.distributions):
            if isinstance(d, _BatchedCategoricalDistributions):
                active_weights = d.weights[active_indices, :]
                rnd_quantile = rng.rand(batch_size)
                cum_probs = np.cumsum(active_weights, axis=-1)
                assert np.isclose(cum_probs[:, -1], 1).all()
                cum_probs[:, -1] = 1  # Avoid numerical errors.
                ret[:, i] = np.sum(cum_probs < rnd_quantile[:, np.newaxis], axis=-1)
            elif isinstance(d, _BatchedTruncNormDistributions):
                numerical_dists.append(d)
                numerical_inds.append(i)
            elif isinstance(d, _BatchedDiscreteTruncNormDistributions):
                disc_inds.append(i)
                numerical_dists.append(d)
                numerical_inds.append(i)
            else:
                assert False

        if len(numerical_dists):
            active_mus = np.asarray([d.mu[active_indices] for d in numerical_dists])
            active_sigmas = np.asarray([d.sigma[active_indices] for d in numerical_dists])
            lows = np.array([d.low for d in numerical_dists])
            highs = np.array([d.high for d in numerical_dists])
            steps = np.array([getattr(d, "step", 0.0) for d in numerical_dists])
            ret[:, numerical_inds] = _truncnorm.rvs(
                a=((lows - steps / 2)[:, np.newaxis] - active_mus) / active_sigmas,
                b=((highs + steps / 2)[:, np.newaxis] - active_mus) / active_sigmas,
                loc=active_mus,
                scale=active_sigmas,
                random_state=rng,
            ).T
            steps_not_0 = np.nonzero(steps != 0.0)[0]
            low_d, step_d, high_d = lows[steps_not_0], steps[steps_not_0], highs[steps_not_0]
            ret[:, disc_inds] = np.clip(
                low_d + np.round((ret[:, disc_inds] - low_d) / step_d) * step_d, low_d, high_d
            )
        return ret

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        weighted_log_pdf = np.zeros((len(x), len(self.weights)), dtype=np.float64)
        cont_dists = []
        cont_inds = []
        for i, d in enumerate(self.distributions):
            if isinstance(d, _BatchedCategoricalDistributions):
                xi = x[:, i, np.newaxis, np.newaxis].astype(np.int64)
                weighted_log_pdf += np.log(np.take_along_axis(d.weights[np.newaxis], xi, axis=-1))[
                    ..., 0
                ]
            elif isinstance(d, _BatchedTruncNormDistributions):
                cont_dists.append(d)
                cont_inds.append(i)
            elif isinstance(d, _BatchedDiscreteTruncNormDistributions):
                xi_uniq, xi_inv = np.unique(x[:, i], return_inverse=True)
                mu_uniq, sigma_uniq, mu_sigma_inv = _unique_inverse_2d(d.mu, d.sigma)
                weighted_log_pdf += _log_gauss_mass_unique(
                    ((xi_uniq - d.step / 2)[:, np.newaxis] - mu_uniq) / sigma_uniq,
                    ((xi_uniq + d.step / 2)[:, np.newaxis] - mu_uniq) / sigma_uniq,
                )[np.ix_(xi_inv, mu_sigma_inv)]
                # Very unlikely to observe duplications below, so we skip the unique operation.
                weighted_log_pdf -= _truncnorm._log_gauss_mass(
                    (d.low - d.step / 2 - mu_uniq) / sigma_uniq,
                    (d.high + d.step / 2 - mu_uniq) / sigma_uniq,
                )[mu_sigma_inv]
            else:
                assert False

        if len(cont_inds):
            mus_cont = np.asarray([d.mu for d in cont_dists]).T
            sigmas_cont = np.asarray([d.sigma for d in cont_dists]).T
            weighted_log_pdf += _truncnorm.logpdf(
                x[:, np.newaxis, cont_inds],
                a=(np.asarray([d.low for d in cont_dists]) - mus_cont) / sigmas_cont,
                b=(np.asarray([d.high for d in cont_dists]) - mus_cont) / sigmas_cont,
                loc=mus_cont,
                scale=sigmas_cont,
            ).sum(axis=-1)

        weighted_log_pdf += np.log(self.weights[np.newaxis])
        max_ = weighted_log_pdf.max(axis=1)
        # We need to avoid (-inf) - (-inf) when the probability is zero.
        max_[np.isneginf(max_)] = 0
        with np.errstate(divide="ignore"):  # Suppress warning in log(0).
            return np.log(np.exp(weighted_log_pdf - max_[:, None]).sum(axis=1)) + max_
