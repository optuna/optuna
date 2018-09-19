from mock import Mock
from mock import patch
import pytest
import typing  # NOQA

from optuna import distributions
from optuna import samplers
from optuna import storages
from optuna.study import create_study
from optuna.trial import Trial

parametrize_storage = pytest.mark.parametrize(
    'storage_init_func',
    [storages.InMemoryStorage, lambda: storages.RDBStorage('sqlite:///:memory:')]
)


@parametrize_storage
def test_suggest_uniform(storage_init_func):
    # type: (typing.Callable[[], storages.BaseStorage]) -> None

    mock = Mock()
    mock.side_effect = [1., 2., 3.]
    sampler = samplers.RandomSampler()

    with patch.object(sampler, 'sample', mock) as mock_object:
        study = create_study(storage_init_func(), sampler=sampler)
        trial = Trial(study, study.storage.create_new_trial_id(study.study_id))
        distribution = distributions.UniformDistribution(low=0., high=3.)

        assert trial._suggest('x', distribution) == 1.  # Test suggesting a param.
        assert trial._suggest('x', distribution) == 1.  # Test suggesting the same param.
        assert trial._suggest('y', distribution) == 3.  # Test suggesting a different param.
        assert trial.params == {'x': 1., 'y': 3.}
        assert mock_object.call_count == 3


@parametrize_storage
def test_suggest_discrete_uniform(storage_init_func):
    # type: (typing.Callable[[], storages.BaseStorage]) -> None

    mock = Mock()
    mock.side_effect = [1., 2., 3.]
    sampler = samplers.RandomSampler()

    with patch.object(sampler, 'sample', mock) as mock_object:
        study = create_study(storage_init_func(), sampler=sampler)
        trial = Trial(study, study.storage.create_new_trial_id(study.study_id))
        distribution = distributions.DiscreteUniformDistribution(low=0., high=3., q=1.)

        assert trial._suggest('x', distribution) == 1.  # Test suggesting a param.
        assert trial._suggest('x', distribution) == 1.  # Test suggesting the same param.
        assert trial._suggest('y', distribution) == 3.  # Test suggesting a different param.
        assert trial.params == {'x': 1., 'y': 3.}
        assert mock_object.call_count == 3


@parametrize_storage
def test_suggest_int(storage_init_func):
    # type: (typing.Callable[[], storages.BaseStorage]) -> None

    mock = Mock()
    mock.side_effect = [1, 2, 3]
    sampler = samplers.RandomSampler()

    with patch.object(sampler, 'sample', mock) as mock_object:
        study = create_study(storage_init_func(), sampler=sampler)
        trial = Trial(study, study.storage.create_new_trial_id(study.study_id))
        distribution = distributions.IntUniformDistribution(low=0, high=3)

        assert trial._suggest('x', distribution) == 1  # Test suggesting a param.
        assert trial._suggest('x', distribution) == 1  # Test suggesting the same param.
        assert trial._suggest('y', distribution) == 3  # Test suggesting a different param.
        assert trial.params == {'x': 1, 'y': 3}
        assert mock_object.call_count == 3
