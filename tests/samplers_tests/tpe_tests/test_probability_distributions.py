import warnings

import numpy as np

from optuna.samplers._tpe.probability_distributions import _BatchedCategoricalDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDiscreteTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _MixtureOfProductDistribution


def test_mixture_of_product_distribution() -> None:
    dist0 = _BatchedTruncNormDistributions(
        mu=np.array([0.2, 3.0]),
        sigma=np.array([0.8, 1.0]),
        low=-1.0,
        high=1.0,
    )
    dist1 = _BatchedDiscreteTruncNormDistributions(
        mu=np.array([0.0, 1.0]),
        sigma=np.array([1.0, 1.0]),
        low=-1.0,
        high=0.5,
        step=0.5,
    )
    dist2 = _BatchedCategoricalDistributions(weights=np.array([[0.4, 0.6], [0.2, 0.8]]))
    mixture_distribution = _MixtureOfProductDistribution(
        weights=np.array([0.5, 0.5]),
        distributions=[dist0, dist1, dist2],
    )
    samples = mixture_distribution.sample(np.random.RandomState(0), 5)

    # Test that the shapes are correct.
    assert samples.shape == (5, 3)

    # Test that the samples are in the valid range.

    assert np.all(dist0.low <= samples[:, 0])
    assert np.all(samples[:, 0] <= dist0.high)
    assert np.all(dist1.low <= samples[:, 1])
    assert np.all(samples[:, 1] <= dist1.high)
    np.testing.assert_almost_equal(
        np.fmod(
            samples[:, 1] - dist1.low,
            dist1.step,
        ),
        0.0,
    )
    assert np.all(0 <= samples[:, 2])
    assert np.all(samples[:, 2] <= 1)
    assert np.all(np.fmod(samples[:, 2], 1.0) == 0.0)

    # Test reproducibility.
    assert np.all(samples == mixture_distribution.sample(np.random.RandomState(0), 5))
    assert not np.all(samples == mixture_distribution.sample(np.random.RandomState(1), 5))

    log_pdf = mixture_distribution.log_pdf(samples)
    assert log_pdf.shape == (5,)


def test_mixture_of_product_distribution_extreme_case() -> None:
    rng = np.random.RandomState(0)
    mixture_distribution = _MixtureOfProductDistribution(
        weights=np.array([1.0, 0.0]),
        distributions=[
            _BatchedTruncNormDistributions(
                mu=np.array([0.5, 0.3]),
                sigma=np.array([1e-10, 1.0]),
                low=-1.0,
                high=1.0,
            ),
            _BatchedDiscreteTruncNormDistributions(
                mu=np.array([-0.5, 1.0]),
                sigma=np.array([1e-10, 1.0]),
                low=-1.0,
                high=0.5,
                step=0.5,
            ),
            _BatchedCategoricalDistributions(weights=np.array([[0, 1], [0.2, 0.8]])),
        ],
    )
    samples = mixture_distribution.sample(rng, 2)
    np.testing.assert_almost_equal(samples, np.array([[0.5, -0.5, 1.0]] * 2))

    # The first point has the highest probability,
    # and all other points have probability almost zero.
    x = np.array([[0.5, 0.5, 1.0], [0.1, 0.5, 1.0], [0.5, 0.0, 1.0], [0.5, 0.5, 0.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # Ignore log(0) warnings.
        log_pdf = mixture_distribution.log_pdf(x)
    assert np.all(log_pdf[1:] < -100)
