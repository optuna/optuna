import typing
import pytest
import numpy as np

import pfnopt


parametrize_sampler = pytest.mark.parametrize(
    'sampler_init_func',
    [pfnopt.samplers.RandomSampler, pfnopt.samplers.TPESampler]
)


@parametrize_sampler
def test_uniform(sampler_init_func):
    # type: (typing.Callable[[], pfnopt.samplers.BaseSampler]) -> None

    storage = pfnopt.storages.get_storage(None)
    study = 0

    sampler = sampler_init_func()
    distribution = pfnopt.distributions.UniformDistribution(-1., 1.)
    points = np.array([sampler.sample(storage, study, 'x', distribution) for _ in range(100)])
    assert np.all(points >= -1.)
    assert np.all(points < 1.)


@parametrize_sampler
def test_quniform(sampler_init_func):
    # type: (typing.Callable[[], pfnopt.samplers.BaseSampler]) -> None

    sampler = sampler_init_func()

    # Test to sample integer value: q = 1
    storage = pfnopt.storages.get_storage(None)
    study = 0
    distribution = pfnopt.distributions.QUniformDistribution(-10., 10., 1.)
    points = np.array([sampler.sample(storage, study, 'x', distribution) for _ in range(100)])
    assert np.all(points >= -10)
    assert np.all(points <= 10)
    round_points = np.round(points)
    np.testing.assert_almost_equal(round_points, points)

    # Test to sample quantized floating point value: [-10.2, 10.2], q = 0.1
    storage = pfnopt.storages.get_storage(None)
    study = 0
    distribution = pfnopt.distributions.QUniformDistribution(-10.2, 10.2, 0.1)
    points = np.array([sampler.sample(storage, study, 'x', distribution) for _ in range(100)])
    assert np.all(points >= -10.2)
    assert np.all(points <= 10.2)
    round_points = np.round(10 * points)
    np.testing.assert_almost_equal(round_points, 10 * points)

