from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import math

from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.samplers._tpe.probability_distributions import _BatchedDiscreteTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDistributions


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from optuna.trial import FrozenTrial


class ScottParzenEstimator(_ParzenEstimator):
    """1D ParzenEstimator using the bandwidth selection by Scott's rule."""

    def _calculate_numerical_distributions(
        self,
        observations: np.ndarray,
        search_space: FloatDistribution | IntDistribution,
        parameters: _ParzenEstimatorParameters,
    ) -> _BatchedDistributions:
        # NOTE: The Optuna TPE bandwidth selection is too wide for this analysis.
        # So use the Scott's rule by Scott, D.W. (1992),
        # Multivariate Density Estimation: Theory, Practice, and Visualization.
        step = search_space.step
        assert step is not None and np.isclose(step, 1.0), "MyPy redefinition."

        n_trials = np.sum(self._counts)
        counts_non_zero = self._counts[self._counts > 0]
        weights = counts_non_zero / n_trials
        mus = np.arange(self.n_steps)[self._counts > 0]
        mean_est = mus @ weights
        sigma_est = np.sqrt((mus - mean_est) ** 2 @ counts_non_zero / max(1, n_trials - 1))

        count_cum = np.cumsum(counts_non_zero)
        idx_q25 = np.searchsorted(count_cum, n_trials // 4, side="left")
        idx_q75 = np.searchsorted(count_cum, n_trials * 3 // 4, side="right")
        interquantile_range = mus[min(mus.size - 1, idx_q75)] - mus[idx_q25]
        sigma_est = 1.059 * min(interquantile_range / 1.34, sigma_est) * n_trials ** (-0.2)
        # To avoid numerical errors. 0.5/1.64 means 1.64sigma (=90%) will fit in the target grid.
        sigma_min = 0.5 / 1.64
        sigmas = np.full_like(mus, max(sigma_est, sigma_min), dtype=np.float64)
        mus = np.append(mus, [0.5 * (search_space.low + search_space.high)])
        sigmas = np.append(sigmas, [1.0 * (search_space.high - search_space.low + 1)])

        return _BatchedDiscreteTruncNormDistributions(
            mu=mus, sigma=sigmas, low=0, high=self.n_steps - 1, step=1
        )

    @property
    def n_steps(self) -> int:
        return self._n_steps

    def pdf(self, samples: np.ndarray) -> np.ndarray:
        return np.exp(self.log_pdf({self._param_name: samples}))




def _count_numerical_param_in_grid(
    param_name: str,
    dist: IntDistribution | FloatDistribution,
    trials: list[FrozenTrial],
    n_steps: int,
) -> np.ndarray:
    assert isinstance(dist, (FloatDistribution, IntDistribution)), "Unexpected distribution."
    if isinstance(dist, IntDistribution) and dist.log:
        log2_domain_size = int(np.ceil(np.log(dist.high - dist.low + 1) / np.log(2))) + 1
        n_steps = min(log2_domain_size, n_steps)
    elif dist.step is not None:
        assert not dist.log, "log must be False when step is not None."
        n_steps = min(round((dist.high - dist.low) / dist.step) + 1, n_steps)
    low, high = (math.log(dist.low), math.log(dist.high)) if dist.log else (dist.low, dist.high)
    param_values = (np.log if dist.log else np.asarray)([t.params[param_name] for t in trials])
    step_size = (high - low) / (n_steps - 1)
    indices = np.minimum(((param_values - low) / step_size).astype(int), n_steps - 1)
    return np.bincount(indices, minlength=n_steps)


def _count_categorical_param_in_grid(
    param_name: str, dist: CategoricalDistribution, trials: list[FrozenTrial]
) -> np.ndarray:
    indices = [int(dist.to_internal_repr(t.params[param_name])) for t in trials]
    return np.bincount(indices, minlength=len(dist.choices))


def build_parzen_estimator_on_grid(
    param_name: str,
    dist: BaseDistribution,
    trials: list[FrozenTrial],
    n_steps: int,
    prior_weight: float,
) -> ScottParzenEstimator:
    rounded_dist: IntDistribution | CategoricalDistribution
    if isinstance(dist, (IntDistribution, FloatDistribution)):
        counts = _count_numerical_param_in_grid(param_name, dist, trials, n_steps)
        rounded_dist = IntDistribution(low=0, high=counts.size - 1)
    elif isinstance(dist, CategoricalDistribution):
        counts = _count_categorical_param_in_grid(param_name, dist, trials)
        rounded_dist = dist
    else:
        assert False, f"Got an unknown dist with the type {type(dist)}."

    observations = np.flatnonzero(counts)
    weights = counts[observations].astype(np.float64)
    parameters = _ParzenEstimatorParameters(
        prior_weight=prior_weight,
        consider_magic_clip=False,
        consider_endpoints=False,
        weights=lambda x: np.empty(0),
        multivariate=True,
        categorical_distance_func={},
    )
    return ScottParzenEstimator(
        {param_name: observations},
        {param_name: rounded_dist},
        parameters,
        weights,
    )
