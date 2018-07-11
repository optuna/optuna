import numpy as np
import pytest
import typing  # NOQA

import pfnopt


parametrize_sampler = pytest.mark.parametrize(
    'sampler_class',
    [pfnopt.samplers.RandomSampler, pfnopt.samplers.TPESampler]
)


@parametrize_sampler
def test_uniform(sampler_class):
    # type: (typing.Callable[[], pfnopt.samplers.BaseSampler]) -> None

    storage = pfnopt.storages.get_storage(None)
    study_id = storage.create_new_study_id()

    sampler = sampler_class()
    distribution = pfnopt.distributions.UniformDistribution(-1., 1.)
    points = np.array([sampler.sample(storage, study_id, 'x', distribution) for _ in range(100)])
    assert np.all(points >= -1.)
    assert np.all(points < 1.)


@parametrize_sampler
def test_discrete_uniform(sampler_class):
    # type: (typing.Callable[[], pfnopt.samplers.BaseSampler]) -> None

    sampler = sampler_class()

    # Test to sample integer value: q = 1
    storage = pfnopt.storages.get_storage(None)
    study_id = storage.create_new_study_id()

    distribution = pfnopt.distributions.DiscreteUniformDistribution(-10., 10., 1.)
    points = np.array([sampler.sample(storage, study_id, 'x', distribution) for _ in range(100)])
    assert np.all(points >= -10)
    assert np.all(points <= 10)
    round_points = np.round(points)
    np.testing.assert_almost_equal(round_points, points)

    # Test to sample quantized floating point value: [-10.2, 10.2], q = 0.1
    distribution = pfnopt.distributions.DiscreteUniformDistribution(-10.2, 10.2, 0.1)
    points = np.array([sampler.sample(storage, study_id, 'y', distribution) for _ in range(100)])
    assert np.all(points >= -10.2)
    assert np.all(points <= 10.2)
    round_points = np.round(10 * points)
    np.testing.assert_almost_equal(round_points, 10 * points)
