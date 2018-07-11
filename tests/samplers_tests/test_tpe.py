import numpy as np

import pfnopt


def test_uniform():
    # type: () -> None

    sampler = pfnopt.samplers.TPESampler(seed=123)
    distribution = pfnopt.distributions.UniformDistribution(-1., 1.)
    points = np.array([sampler._sample_uniform(
        distribution=distribution, below=[], above=[]) for i in range(100)])
    assert np.all(points >= -1.)
    assert np.all(points < 1.)


def test_quniform():
    # type: () -> None

    sampler = pfnopt.samplers.TPESampler(seed=123)
    distribution = pfnopt.distributions.QUniformDistribution(-10.5, 10.5, 1)
    points = np.array([sampler._sample_quniform(
        distribution=distribution, below=[], above=[]) for i in range(100)])
    assert np.all(points >= -10)
    assert np.all(points <= 10)
    round_points = np.round(points)
    np.testing.assert_almost_equal(round_points, points)

    distribution = pfnopt.distributions.QUniformDistribution(-10.05, 10.05, 0.1)
    points = np.array([sampler._sample_quniform(
        distribution=distribution, below=[], above=[]) for i in range(100)])
    assert np.all(points >= -10)
    assert np.all(points <= 10)
    round_points = np.round(10 * points)
    np.testing.assert_almost_equal(round_points, 10 * points)
