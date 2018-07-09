import numpy as np

import pfnopt


def test_uniform():
    # type: () -> None

    storage = pfnopt.storages.get_storage(None)
    study = 1

    sampler = pfnopt.samplers.RandomSampler(seed=123)
    distribution = pfnopt.distributions.UniformDistribution(-1., 1.)
    points = np.array([sampler.sample(storage, study, 'x', distribution) for i in range(100)])
    assert np.all(points > -1.)
    assert np.all(points < 1.)


def test_quniform():
    # type: () -> None

    storage = pfnopt.storages.get_storage(None)
    study = 1

    sampler = pfnopt.samplers.RandomSampler(seed=123)
    distribution = pfnopt.distributions.QUniformDistribution(-10., 10., 1)
    points = np.array([sampler.sample(storage, study, 'x', distribution) for i in range(100)])
    assert np.all(points >= -10)
    assert np.all(points <= 10)
    round_points = np.round(points)
    np.testing.assert_almost_equal(round_points, points)

    distribution = pfnopt.distributions.QUniformDistribution(-10., 10., 0.1)
    points = np.array([sampler.sample(storage, study, 'x', distribution) for i in range(100)])
    assert np.all(points >= -10)
    assert np.all(points <= 10)
    round_points = np.round(10 * points)
    np.testing.assert_almost_equal(round_points, 10 * points)
