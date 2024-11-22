from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from optuna import distributions
from optuna.distributions import CategoricalChoiceType
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters
from optuna.samplers._tpe.probability_distributions import _BatchedCategoricalDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDiscreteTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _MixtureOfProductDistribution
from optuna.samplers._tpe.sampler import default_weights


def assert_distribution_almost_equal(
    d1: _MixtureOfProductDistribution, d2: _MixtureOfProductDistribution
) -> None:
    np.testing.assert_almost_equal(d1.weights, d2.weights)
    for d1_, d2_ in zip(d1.distributions, d2.distributions):
        assert type(d1_) is type(d2_)
        for field1, field2 in zip(d1_, d2_):
            np.testing.assert_almost_equal(np.array(field1), np.array(field2))


SEARCH_SPACE = {
    "a": distributions.FloatDistribution(1.0, 100.0),
    "b": distributions.FloatDistribution(1.0, 100.0, log=True),
    "c": distributions.FloatDistribution(1.0, 100.0, step=3.0),
    "d": distributions.IntDistribution(1, 100),
    "e": distributions.IntDistribution(1, 100, log=True),
    "f": distributions.CategoricalDistribution(["x", "y", "z"]),
    "g": distributions.CategoricalDistribution([0.0, float("inf"), float("nan"), None]),
}

MULTIVARIATE_SAMPLES = {
    "a": np.array([1.0]),
    "b": np.array([1.0]),
    "c": np.array([1.0]),
    "d": np.array([1]),
    "e": np.array([1]),
    "f": np.array([1]),
    "g": np.array([1]),
}


@pytest.mark.parametrize("consider_prior", [True, False])
@pytest.mark.parametrize("multivariate", [True, False])
def test_init_parzen_estimator(consider_prior: bool, multivariate: bool) -> None:
    parameters = _ParzenEstimatorParameters(
        consider_prior=consider_prior,
        prior_weight=1.0,
        consider_magic_clip=False,
        consider_endpoints=False,
        weights=lambda x: np.arange(x) + 1.0,
        multivariate=multivariate,
        categorical_distance_func={},
    )

    mpe = _ParzenEstimator(MULTIVARIATE_SAMPLES, SEARCH_SPACE, parameters)

    weights = np.array([1] + consider_prior * [1], dtype=float)
    weights /= weights.sum()

    expected_univariate = _MixtureOfProductDistribution(
        weights=weights,
        distributions=[
            _BatchedTruncNormDistributions(
                mu=np.array([1.0] + consider_prior * [50.5]),
                sigma=np.array([49.5 if consider_prior else 99.0] + consider_prior * [99.0]),
                low=1.0,
                high=100.0,
            ),
            _BatchedTruncNormDistributions(
                mu=np.array([np.log(1.0)] + consider_prior * [np.log(100) / 2.0]),
                sigma=np.array(
                    [np.log(100) / 2 if consider_prior else np.log(100.0)]
                    + consider_prior * [np.log(100)]
                ),
                low=np.log(1.0),
                high=np.log(100.0),
            ),
            _BatchedDiscreteTruncNormDistributions(
                mu=np.array([1.0] + consider_prior * [50.5]),
                sigma=np.array([49.5 if consider_prior else 100.5] + consider_prior * [102.0]),
                low=1.0,
                high=100.0,
                step=3.0,
            ),
            _BatchedDiscreteTruncNormDistributions(
                mu=np.array([1.0] + consider_prior * [50.5]),
                sigma=np.array([49.5 if consider_prior else 99.5] + consider_prior * [100.0]),
                low=1,
                high=100,
                step=1,
            ),
            _BatchedTruncNormDistributions(
                mu=np.array(
                    [np.log(1.0)] + consider_prior * [(np.log(100.5) + np.log(0.5)) / 2.0]
                ),
                sigma=np.array(
                    [(np.log(100.5) + np.log(0.5)) / 2 if consider_prior else np.log(100.5)]
                    + consider_prior * [np.log(100.5) - np.log(0.5)]
                ),
                low=np.log(0.5),
                high=np.log(100.5),
            ),
            _BatchedCategoricalDistributions(
                np.array([[0.2, 0.6, 0.2], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
                if consider_prior
                else np.array([[0.25, 0.5, 0.25]])
            ),
            _BatchedCategoricalDistributions(
                np.array(
                    [
                        [1.0 / 6.0, 0.5, 1.0 / 6.0, 1.0 / 6.0],
                        [1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0],
                    ]
                )
                if consider_prior
                else np.array([[0.2, 0.4, 0.2, 0.2]])
            ),
        ],
    )
    SIGMA0 = 0.2
    expected_multivarite = _MixtureOfProductDistribution(
        weights=weights,
        distributions=[
            _BatchedTruncNormDistributions(
                mu=np.array([1.0] + consider_prior * [50.5]),
                sigma=np.array([SIGMA0 * 99.0] + consider_prior * [99.0]),
                low=1.0,
                high=100.0,
            ),
            _BatchedTruncNormDistributions(
                mu=np.array([np.log(1.0)] + consider_prior * [np.log(100) / 2.0]),
                sigma=np.array([SIGMA0 * np.log(100)] + consider_prior * [np.log(100)]),
                low=np.log(1.0),
                high=np.log(100.0),
            ),
            _BatchedDiscreteTruncNormDistributions(
                mu=np.array([1.0] + consider_prior * [50.5]),
                sigma=np.array([SIGMA0 * 102.0] + consider_prior * [102.0]),
                low=1.0,
                high=100.0,
                step=3.0,
            ),
            _BatchedDiscreteTruncNormDistributions(
                mu=np.array([1.0] + consider_prior * [50.5]),
                sigma=np.array([SIGMA0 * 100.0] + consider_prior * [100.0]),
                low=1,
                high=100,
                step=1,
            ),
            _BatchedTruncNormDistributions(
                mu=np.array(
                    [np.log(1.0)] + consider_prior * [(np.log(100.5) + np.log(0.5)) / 2.0]
                ),
                sigma=np.array(
                    [SIGMA0 * (np.log(100.5) - np.log(0.5))]
                    + consider_prior * [np.log(100.5) - np.log(0.5)]
                ),
                low=np.log(0.5),
                high=np.log(100.5),
            ),
            _BatchedCategoricalDistributions(
                np.array([[0.2, 0.6, 0.2], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
                if consider_prior
                else np.array([[0.25, 0.5, 0.25]])
            ),
            _BatchedCategoricalDistributions(
                np.array(
                    [
                        [1.0 / 6.0, 0.5, 1.0 / 6.0, 1.0 / 6.0],
                        [1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0],
                    ]
                    if consider_prior
                    else np.array([[0.2, 0.4, 0.2, 0.2]])
                )
            ),
        ],
    )

    expected = expected_multivarite if multivariate else expected_univariate

    # Test that the distribution is correct.
    assert_distribution_almost_equal(mpe._mixture_distribution, expected)

    # Test that the sampled values are valid.
    samples = mpe.sample(np.random.RandomState(0), 10)
    for param, values in samples.items():
        for value in values:
            assert SEARCH_SPACE[param]._contains(value)


@pytest.mark.parametrize("mus", (np.asarray([]), np.asarray([0.4]), np.asarray([-0.4, 0.4])))
@pytest.mark.parametrize("prior_weight", [1.0, 0.01, 100.0])
@pytest.mark.parametrize("prior", (True, False))
@pytest.mark.parametrize("magic_clip", (True, False))
@pytest.mark.parametrize("endpoints", (True, False))
@pytest.mark.parametrize("multivariate", (True, False))
def test_calculate_shape_check(
    mus: np.ndarray,
    prior_weight: float,
    prior: bool,
    magic_clip: bool,
    endpoints: bool,
    multivariate: bool,
) -> None:
    parameters = _ParzenEstimatorParameters(
        prior_weight=prior_weight,
        consider_prior=prior,
        consider_magic_clip=magic_clip,
        consider_endpoints=endpoints,
        weights=default_weights,
        multivariate=multivariate,
        categorical_distance_func={},
    )
    mpe = _ParzenEstimator(
        {"a": mus}, {"a": distributions.FloatDistribution(-1.0, 1.0)}, parameters
    )
    assert len(mpe._mixture_distribution.weights) == max(len(mus) + int(prior), 1)


@pytest.mark.parametrize("mus", (np.asarray([]), np.asarray([0.4]), np.asarray([-0.4, 0.4])))
@pytest.mark.parametrize("prior_weight", [1.0, 0.01, 100.0])
@pytest.mark.parametrize("prior", (True, False))
@pytest.mark.parametrize("categorical_distance_func", ({}, {"c": lambda x, y: abs(x - y)}))
def test_calculate_shape_check_categorical(
    mus: np.ndarray,
    prior_weight: float,
    prior: bool,
    categorical_distance_func: dict[
        str,
        Callable[[CategoricalChoiceType, CategoricalChoiceType], float],
    ],
) -> None:
    parameters = _ParzenEstimatorParameters(
        prior_weight=prior_weight,
        consider_prior=prior,
        consider_magic_clip=True,
        consider_endpoints=False,
        weights=default_weights,
        multivariate=False,
        categorical_distance_func=categorical_distance_func,
    )
    mpe = _ParzenEstimator(
        {"c": mus}, {"c": distributions.CategoricalDistribution([0.0, 1.0, 2.0])}, parameters
    )
    assert len(mpe._mixture_distribution.weights) == max(len(mus) + int(prior), 1)


@pytest.mark.parametrize("prior_weight", [None, -1.0, 0.0])
@pytest.mark.parametrize("mus", (np.asarray([]), np.asarray([0.4]), np.asarray([-0.4, 0.4])))
def test_invalid_prior_weight(prior_weight: float, mus: np.ndarray) -> None:
    parameters = _ParzenEstimatorParameters(
        prior_weight=prior_weight,
        consider_prior=True,
        consider_magic_clip=False,
        consider_endpoints=False,
        weights=default_weights,
        multivariate=False,
        categorical_distance_func={},
    )
    with pytest.raises(ValueError):
        _ParzenEstimator({"a": mus}, {"a": distributions.FloatDistribution(-1.0, 1.0)}, parameters)


# TODO(ytsmiling): Improve test coverage for weights.
@pytest.mark.parametrize(
    "mus, flags, expected",
    [
        [
            np.asarray([]),
            {"prior": False, "magic_clip": False, "endpoints": True},
            {"weights": [1.0], "mus": [0.0], "sigmas": [2.0]},
        ],
        [
            np.asarray([]),
            {"prior": True, "magic_clip": False, "endpoints": True},
            {"weights": [1.0], "mus": [0.0], "sigmas": [2.0]},
        ],
        [
            np.asarray([0.4]),
            {"prior": True, "magic_clip": False, "endpoints": True},
            {"weights": [0.5, 0.5], "mus": [0.4, 0.0], "sigmas": [0.6, 2.0]},
        ],
        [
            np.asarray([-0.4]),
            {"prior": True, "magic_clip": False, "endpoints": True},
            {"weights": [0.5, 0.5], "mus": [-0.4, 0.0], "sigmas": [0.6, 2.0]},
        ],
        [
            np.asarray([-0.4, 0.4]),
            {"prior": True, "magic_clip": False, "endpoints": True},
            {"weights": [1.0 / 3] * 3, "mus": [-0.4, 0.4, 0.0], "sigmas": [0.6, 0.6, 2.0]},
        ],
        [
            np.asarray([-0.4, 0.4]),
            {"prior": True, "magic_clip": False, "endpoints": False},
            {"weights": [1.0 / 3] * 3, "mus": [-0.4, 0.4, 0.0], "sigmas": [0.4, 0.4, 2.0]},
        ],
        [
            np.asarray([-0.4, 0.4]),
            {"prior": False, "magic_clip": False, "endpoints": True},
            {"weights": [0.5, 0.5], "mus": [-0.4, 0.4], "sigmas": [0.8, 0.8]},
        ],
        [
            np.asarray([-0.4, 0.4, 0.41, 0.42]),
            {"prior": False, "magic_clip": False, "endpoints": True},
            {
                "weights": [0.25, 0.25, 0.25, 0.25],
                "mus": [-0.4, 0.4, 0.41, 0.42],
                "sigmas": [0.8, 0.8, 0.01, 0.58],
            },
        ],
        [
            np.asarray([-0.4, 0.4, 0.41, 0.42]),
            {"prior": False, "magic_clip": True, "endpoints": True},
            {
                "weights": [0.25, 0.25, 0.25, 0.25],
                "mus": [-0.4, 0.4, 0.41, 0.42],
                "sigmas": [0.8, 0.8, 0.4, 0.58],
            },
        ],
    ],
)
def test_calculate(
    mus: np.ndarray, flags: dict[str, bool], expected: dict[str, list[float]]
) -> None:
    parameters = _ParzenEstimatorParameters(
        prior_weight=1.0,
        consider_prior=flags["prior"],
        consider_magic_clip=flags["magic_clip"],
        consider_endpoints=flags["endpoints"],
        weights=default_weights,
        multivariate=False,
        categorical_distance_func={},
    )
    mpe = _ParzenEstimator(
        {"a": mus}, {"a": distributions.FloatDistribution(-1.0, 1.0)}, parameters
    )
    expected_distribution = _MixtureOfProductDistribution(
        weights=np.asarray(expected["weights"]),
        distributions=[
            _BatchedTruncNormDistributions(
                mu=np.asarray(expected["mus"]),
                sigma=np.asarray(expected["sigmas"]),
                low=-1.0,
                high=1.0,
            )
        ],
    )
    assert_distribution_almost_equal(mpe._mixture_distribution, expected_distribution)


@pytest.mark.parametrize(
    "weights",
    [
        lambda x: np.zeros(x),
        lambda x: -np.ones(x),
        lambda x: float("inf") * np.ones(x),
        lambda x: -float("inf") * np.ones(x),
        lambda x: np.asarray([float("nan") for _ in range(x)]),
    ],
)
def test_invalid_weights(weights: Callable[[int], np.ndarray]) -> None:
    parameters = _ParzenEstimatorParameters(
        prior_weight=1.0,
        consider_prior=False,
        consider_magic_clip=False,
        consider_endpoints=False,
        weights=weights,
        multivariate=False,
        categorical_distance_func={},
    )
    with pytest.raises(ValueError):
        _ParzenEstimator(
            {"a": np.asarray([0.0])}, {"a": distributions.FloatDistribution(-1.0, 1.0)}, parameters
        )
