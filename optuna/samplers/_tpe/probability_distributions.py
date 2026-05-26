from __future__ import annotations

import math
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

    @property
    def adapted_low(self) -> float:
        return self.low

    @property
    def adapted_high(self) -> float:
        return self.high

    @property
    def is_log(self) -> bool:
        return False

    @property
    def step(self) -> float:
        return 0.0


class _BatchedTruncLogNormDistributions(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray
    low: float  # Currently, low and high do not change per trial.
    high: float

    @property
    def adapted_low(self) -> float:
        return math.log(self.low)

    @property
    def adapted_high(self) -> float:
        return math.log(self.high)

    @property
    def is_log(self) -> bool:
        return True

    @property
    def step(self) -> float:
        return 0.0


class _BatchedDiscreteTruncNormDistributions(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray
    low: float  # Currently, low, high and step do not change per trial.
    high: float
    step: float

    @property
    def adapted_low(self) -> float:
        return self.low - self.step / 2

    @property
    def adapted_high(self) -> float:
        return self.high + self.step / 2

    @property
    def is_log(self) -> bool:
        return False


class _BatchedDiscreteTruncLogNormDistributions(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray
    low: float  # Currently, low, high and step do not change per trial.
    high: float
    step: float

    @property
    def adapted_low(self) -> float:
        return math.log(self.low - self.step / 2)

    @property
    def adapted_high(self) -> float:
        return math.log(self.high + self.step / 2)

    @property
    def is_log(self) -> bool:
        return True


_BatchedDistributions = Union[
    _BatchedCategoricalDistributions,
    _BatchedTruncNormDistributions,
    _BatchedTruncLogNormDistributions,
    _BatchedDiscreteTruncNormDistributions,
    _BatchedDiscreteTruncLogNormDistributions,
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
        disc_inds, numerical_inds, log_inds = [], [], []
        numerical_dists = []
        disc_dists = []
        for i, d in enumerate(self.distributions):
            if isinstance(d, _BatchedCategoricalDistributions):
                active_weights = d.weights[active_indices, :]
                rnd_quantile = rng.rand(batch_size)
                cum_probs = np.cumsum(active_weights, axis=-1)
                assert np.isclose(cum_probs[:, -1], 1).all()
                cum_probs[:, -1] = 1  # Avoid numerical errors.
                ret[:, i] = np.sum(cum_probs < rnd_quantile[:, np.newaxis], axis=-1)
            else:
                numerical_dists.append(d)
                numerical_inds.append(i)
                if d.step != 0.0:
                    disc_inds.append(i)
                    disc_dists.append(d)
                if d.is_log:
                    log_inds.append(i)

        if len(numerical_dists):
            active_mus = np.asarray([d.mu[active_indices] for d in numerical_dists])
            active_sigmas = np.asarray([d.sigma[active_indices] for d in numerical_dists])
            adapted_lows = np.asarray([d.adapted_low for d in numerical_dists])
            adapted_highs = np.asarray([d.adapted_high for d in numerical_dists])
            ret[:, numerical_inds] = _truncnorm.rvs(
                a=(adapted_lows[:, np.newaxis] - active_mus) / active_sigmas,
                b=(adapted_highs[:, np.newaxis] - active_mus) / active_sigmas,
                loc=active_mus,
                scale=active_sigmas,
                random_state=rng,
            ).T
            ret[:, log_inds] = np.exp(ret[:, log_inds])
            step_d = np.asarray([d.step for d in disc_dists])
            low_d = np.asarray([d.low for d in disc_dists])
            high_d = np.asarray([d.high for d in disc_dists])
            ret[:, disc_inds] = np.clip(
                low_d + np.round((ret[:, disc_inds] - low_d) / step_d) * step_d, low_d, high_d
            )

        return ret

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        weighted_log_pdf = np.zeros((len(x), len(self.weights)), dtype=np.float64)
        cont_dists = []
        x_cont = []
        lows_cont = []
        highs_cont = []
        for i, d in enumerate(self.distributions):
            if isinstance(d, _BatchedCategoricalDistributions):
                xi = x[:, i, np.newaxis, np.newaxis].astype(np.int64)
                weighted_log_pdf += np.log(np.take_along_axis(d.weights[np.newaxis], xi, axis=-1))[
                    ..., 0
                ]
                continue
            is_log = d.is_log
            if (step := d.step) == 0.0:
                cont_dists.append(d)
                x_cont.append(np.log(x[:, i]) if is_log else x[:, i])
                lows_cont.append(d.adapted_low)
                highs_cont.append(d.adapted_high)
            else:
                xi_uniq, xi_inv = np.unique(x[:, i], return_inverse=True)
                mu_uniq, sigma_uniq, mu_sigma_inv = _unique_inverse_2d(d.mu, d.sigma)
                left = np.log(xi_uniq - step / 2) if is_log else (xi_uniq - step / 2)
                right = np.log(xi_uniq + step / 2) if is_log else (xi_uniq + step / 2)
                weighted_log_pdf += _log_gauss_mass_unique(
                    (left[:, np.newaxis] - mu_uniq) / sigma_uniq,
                    (right[:, np.newaxis] - mu_uniq) / sigma_uniq,
                )[np.ix_(xi_inv, mu_sigma_inv)]
                # Very unlikely to observe duplications below, so we skip the unique operation.
                weighted_log_pdf -= _truncnorm._log_gauss_mass(
                    (d.adapted_low - mu_uniq) / sigma_uniq, (d.adapted_high - mu_uniq) / sigma_uniq
                )[mu_sigma_inv]

        if len(x_cont):
            mus_cont = np.asarray([d.mu for d in cont_dists]).T
            sigmas_cont = np.asarray([d.sigma for d in cont_dists]).T
            weighted_log_pdf += _truncnorm.logpdf(
                np.asarray(x_cont).T[:, np.newaxis, :],
                a=(np.asarray(lows_cont) - mus_cont) / sigmas_cont,
                b=(np.asarray(highs_cont) - mus_cont) / sigmas_cont,
                loc=mus_cont,
                scale=sigmas_cont,
            ).sum(axis=-1)

        weighted_log_pdf += np.log(self.weights[np.newaxis])
        max_ = weighted_log_pdf.max(axis=1)
        # We need to avoid (-inf) - (-inf) when the probability is zero.
        max_[np.isneginf(max_)] = 0
        with np.errstate(divide="ignore"):  # Suppress warning in log(0).
            return np.log(np.exp(weighted_log_pdf - max_[:, None]).sum(axis=1)) + max_
