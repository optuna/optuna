from __future__ import annotations

import numpy as np
import pytest

from optuna.distributions import CategoricalDistribution
from optuna.distributions import IntDistribution
from optuna.importance._ped_anova.scott_parzen_estimator import _ScottParzenEstimator
from optuna.samplers._tpe.probability_distributions import _BatchedCategoricalDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDiscreteTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _MixtureOfProductDistribution
from tests.samplers_tests.tpe_tests.test_parzen_estimator import assert_distribution_almost_equal


DIST_TYPES = ["int", "cat"]


@pytest.mark.parametrize("dist_type", DIST_TYPES)
def test_init_scott_parzen_estimator(dist_type: str) -> None:
    counts = np.array([1, 1, 1, 1]).astype(float)
    is_cat = dist_type == "cat"
    pe = _ScottParzenEstimator(
        param_name="a",
        dist=(
            IntDistribution(low=0, high=counts.size - 1)
            if not is_cat
            else CategoricalDistribution(choices=["a" * i for i in range(counts.size)])
        ),
        counts=counts,
        consider_prior=False,
        prior_weight=0.0,
    )
    assert len(pe._mixture_distribution.distributions) == 1
    assert pe.n_steps == counts.size
    target_pe = pe._mixture_distribution.distributions[0]
    if is_cat:
        assert isinstance(target_pe, _BatchedCategoricalDistributions)
    else:
        assert isinstance(target_pe, _BatchedDiscreteTruncNormDistributions)


@pytest.mark.parametrize(
    "counts,mu,sigma,weights",
    [
        (np.array([0, 0, 0, 1]), np.array([3]), np.array([0.304878]), np.array([1.0])),
        (np.array([0, 0, 100, 0]), np.array([2]), np.array([0.304878]), np.array([1.0])),
        (np.array([1, 2, 3, 4]), np.arange(4), np.array([0.7043276] * 4), (np.arange(4) + 1) / 10),
        (
            np.array([90, 0, 0, 90]),
            np.array([0, 3]),
            np.array([0.5638226] * 2),
            np.array([0.5] * 2),
        ),
        (np.array([1, 0, 0, 1]), np.array([0, 3]), np.array([1.9556729] * 2), np.array([0.5] * 2)),
    ],
)
def test_build_int_scott_parzen_estimator(
    counts: np.ndarray, mu: np.ndarray, sigma: np.ndarray, weights: np.ndarray
) -> None:
    _counts = counts.astype(float)
    pe = _ScottParzenEstimator(
        param_name="a",
        dist=IntDistribution(low=0, high=_counts.size - 1),
        counts=_counts,
        consider_prior=False,
        prior_weight=0.0,
    )
    dist = _BatchedDiscreteTruncNormDistributions(
        mu=mu, sigma=sigma, low=0, high=_counts.size - 1, step=1
    )
    expected_dist = _MixtureOfProductDistribution(weights=weights, distributions=[dist])
    assert_distribution_almost_equal(pe._mixture_distribution, expected_dist)


@pytest.mark.parametrize(
    "counts,weights",
    [
        (np.array([0, 0, 0, 1]), np.array([1.0])),
        (np.array([0, 0, 100, 0]), np.array([1.0])),
        (np.array([1, 2, 3, 4]), (np.arange(4) + 1) / 10),
        (np.array([90, 0, 0, 90]), np.array([0.5] * 2)),
        (np.array([1, 0, 0, 1]), np.array([0.5] * 2)),
    ],
)
def test_build_cat_scott_parzen_estimator(counts: np.ndarray, weights: np.ndarray) -> None:
    _counts = counts.astype(float)
    pe = _ScottParzenEstimator(
        param_name="a",
        dist=CategoricalDistribution(choices=["a" * i for i in range(counts.size)]),
        counts=_counts,
        consider_prior=False,
        prior_weight=0.0,
    )
    dist = _BatchedCategoricalDistributions(weights=np.identity(counts.size)[counts > 0.0])
    expected_dist = _MixtureOfProductDistribution(weights=weights, distributions=[dist])
    assert_distribution_almost_equal(pe._mixture_distribution, expected_dist)
