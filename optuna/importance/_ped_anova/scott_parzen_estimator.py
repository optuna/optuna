from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

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

    def __init__(
        self,
        observations: dict[str, np.ndarray],
        search_space: dict[str, BaseDistribution],
        parameters: _ParzenEstimatorParameters,
        predetermined_weights: np.ndarray | None = None,
    ) -> None:
        assert predetermined_weights is not None
        self._weights = predetermined_weights
        super().__init__(observations, search_space, parameters, predetermined_weights)

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

        low, high = search_space.low, search_space.high
        weights_cum = np.cumsum(self._weights)
        weights_sum = weights_cum[-1]
        if search_space.log:
            observations = np.log(observations)
            low, high = math.log(low), math.log(high)

        mean_est = (observations @ self._weights) / weights_sum
        sigma_est = np.sqrt(
            ((observations - mean_est) ** 2 @ self._weights) / max(1, weights_sum - 1)
        )

        q1_idx = np.searchsorted(weights_cum, weights_sum // 4, side="left")
        q3_idx = np.searchsorted(weights_cum, weights_sum * 3 // 4, side="right")
        iqr = observations[min(observations.size - 1, q3_idx)] - observations[q1_idx]
        sigma_est = 1.059 * min(iqr / 1.34, sigma_est) * weights_sum**-0.2
        # To avoid numerical errors. 0.5/1.64 means 1.64sigma (=90%) will fit in the target grid.
        sigma_min = 0.5 / 1.64
        mus_with_prior = np.r_[observations, (low + high) / 2.0]
        sigmas = np.full_like(observations, max(sigma_est, sigma_min), dtype=np.float64)
        sigmas_with_prior = np.r_[sigmas, high - low + 1]

        return _BatchedDiscreteTruncNormDistributions(
            mu=mus_with_prior, sigma=sigmas_with_prior, low=low, high=high, step=1
        )


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
    # For backward compatibility, midpoint ties go to the lower grid.
    indices = np.ceil((param_values - low) / step_size - 0.5).astype(int)
    indices = np.clip(indices, 0, n_steps - 1)
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
) -> tuple[ScottParzenEstimator, int]:
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
    weights = counts[observations]
    parameters = _ParzenEstimatorParameters(
        prior_weight=prior_weight,
        consider_magic_clip=False,
        consider_endpoints=False,
        weights=lambda x: np.empty(0),
        multivariate=True,
        categorical_distance_func={},
    )
    pe = ScottParzenEstimator(
        {param_name: observations},
        {param_name: rounded_dist},
        parameters,
        weights,
    )
    return pe, counts.size
