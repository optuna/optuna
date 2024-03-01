from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from optuna import create_trial
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.importance._ped_anova.scott_parzen_estimator import _build_parzen_estimator
from optuna.importance._ped_anova.scott_parzen_estimator import _count_categorical_param_in_grid
from optuna.importance._ped_anova.scott_parzen_estimator import _count_numerical_param_in_grid
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
        # NOTE: sigma could change depending on sigma_min picked by heuristic.
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


@pytest.mark.parametrize(
    "dist,params,expected_outcome",
    [
        (IntDistribution(low=-5, high=5), [-5, -5, 1, 5, 5], [2, 0, 1, 0, 2]),
        (IntDistribution(low=1, high=8, log=True), list(range(1, 9)), [1, 1, 3, 3]),
        (FloatDistribution(low=-5.0, high=5.0), np.linspace(-5, 5, 100), [13, 25, 24, 25, 13]),
        (
            FloatDistribution(low=1, high=8, log=True),
            [float(i) for i in range(1, 9)],
            [1, 1, 1, 3, 2],
        ),
    ],
)
def test_count_numerical_param_in_grid(
    dist: IntDistribution | FloatDistribution,
    params: list[int] | list[float],
    expected_outcome: list[int],
) -> None:
    trials = [create_trial(value=0.0, params={"a": p}, distributions={"a": dist}) for p in params]
    res = _count_numerical_param_in_grid(param_name="a", dist=dist, trials=trials, n_steps=5)
    assert np.all(np.asarray(expected_outcome) == res), res


def test_count_categorical_param_in_grid() -> None:
    params = ["a", "b", "a", "d", "a", "a", "d"]
    dist = CategoricalDistribution(choices=["a", "b", "c", "d"])
    expected_outcome = [4, 1, 0, 2]
    trials = [create_trial(value=0.0, params={"a": p}, distributions={"a": dist}) for p in params]
    res = _count_categorical_param_in_grid(param_name="a", dist=dist, trials=trials)
    assert np.all(np.asarray(expected_outcome) == res)


@pytest.mark.parametrize(
    "dist,params",
    [
        (IntDistribution(low=-5, high=5), [1, 2, 3]),
        (IntDistribution(low=1, high=8, log=True), [1, 2, 4, 8]),
        (IntDistribution(low=-5, high=5, step=2), [1, 3, 5]),
        (FloatDistribution(low=-5.0, high=5.0), [1.0, 2.0, 3.0]),
        (FloatDistribution(low=1.0, high=8.0, log=True), [1.0, 2.0, 8.0]),
        (FloatDistribution(low=-5.0, high=5.0, step=0.5), [1.0, 2.0, 3.0]),
        (CategoricalDistribution(choices=["a", "b", "c"]), ["a", "b", "b"]),
    ],
)
def test_build_parzen_estimator(
    dist: BaseDistribution,
    params: list[int] | list[float] | list[str],
) -> None:
    trials = [create_trial(value=0.0, params={"a": p}, distributions={"a": dist}) for p in params]
    pe = _build_parzen_estimator(
        param_name="a",
        dist=dist,
        trials=trials,
        n_steps=50,
        consider_prior=True,
        prior_weight=1.0,
    )
    if isinstance(dist, (IntDistribution, FloatDistribution)):
        assert isinstance(
            pe._mixture_distribution.distributions[0], _BatchedDiscreteTruncNormDistributions
        )
    elif isinstance(dist, CategoricalDistribution):
        assert isinstance(
            pe._mixture_distribution.distributions[0], _BatchedCategoricalDistributions
        )
    else:
        assert False, "Should not be reached."


def test_assert_in_build_parzen_estimator() -> None:
    class UnknownDistribution(BaseDistribution):
        def to_internal_repr(self, param_value_in_external_repr: Any) -> float:
            raise NotImplementedError

        def single(self) -> bool:
            raise NotImplementedError

        def _contains(self, param_value_in_internal_repr: float) -> bool:
            raise NotImplementedError

    with pytest.raises(AssertionError):
        _build_parzen_estimator(
            param_name="a",
            dist=UnknownDistribution(),
            trials=[],
            n_steps=50,
            consider_prior=True,
            prior_weight=1.0,
        )
