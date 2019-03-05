import math
from mock import Mock
from mock import patch
import pytest

from optuna import distributions
from optuna import samplers
from optuna import storages
from optuna.study import create_study
from optuna.trial import FixedTrial
from optuna.trial import Trial
from optuna import types

if types.TYPE_CHECKING:
    import typing  # NOQA

parametrize_storage = pytest.mark.parametrize(
    'storage_init_func',
    [storages.InMemoryStorage, lambda: storages.RDBStorage('sqlite:///:memory:')])


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
@pytest.mark.parametrize(
    'range_config',
    [
        {
            'low': 0.,
            'high': 10.,
            'q': 3.,
            'mod_high': 9.
        },
        {
            'low': 1.,
            'high': 11.,
            'q': 3.,
            'mod_high': 10.
        },
        {
            'low': 64.,
            'high': 1312.,
            'q': 160.,
            'mod_high': 1184.
        },
        # high is excluded due to the round-off error of 10 // 0.1.
        {
            'low': 0.,
            'high': 10.,
            'q': 0.1,
            'mod_high': 9.9
        },
        # high is excluded doe to the round-off error of 10.1 // 0.1
        {
            'low': 0.,
            'high': 10.1,
            'q': 0.1,
            'mod_high': 10.
        },
        {
            'low': 0.,
            'high': 10.,
            'q': math.pi,
            'mod_high': 3 * math.pi
        }
    ])
def test_suggest_discrete_uniform_range(storage_init_func, range_config):
    # type: (typing.Callable[[], storages.BaseStorage], typing.Dict[str, float]) -> None

    sampler = samplers.RandomSampler()

    # Check upper endpoints.
    mock = Mock()
    mock.side_effect = lambda storage, study_id, param_name, distribution: distribution.high
    with patch.object(sampler, 'sample', mock) as mock_object:
        study = create_study(storage_init_func(), sampler=sampler)
        trial = Trial(study, study.storage.create_new_trial_id(study.study_id))

        x = trial.suggest_discrete_uniform('x', range_config['low'], range_config['high'],
                                           range_config['q'])
        assert x == range_config['mod_high']
        assert mock_object.call_count == 1

    # Check lower endpoints.
    mock = Mock()
    mock.side_effect = lambda storage, study_id, param_name, distribution: distribution.low
    with patch.object(sampler, 'sample', mock) as mock_object:
        study = create_study(storage_init_func(), sampler=sampler)
        trial = Trial(study, study.storage.create_new_trial_id(study.study_id))

        x = trial.suggest_discrete_uniform('x', range_config['low'], range_config['high'],
                                           range_config['q'])
        assert x == range_config['low']
        assert mock_object.call_count == 1


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


def test_fixed_trial_suggest_uniform():
    # type: () -> None

    trial = FixedTrial({'x': 1.})
    assert trial.suggest_uniform('x', -100., 100.) == 1.

    with pytest.raises(ValueError):
        trial.suggest_uniform('y', -100., 100.)


def test_fixed_trial_suggest_loguniform():
    # type: () -> None

    trial = FixedTrial({'x': 1.})
    assert trial.suggest_loguniform('x', 0., 1.) == 1.

    with pytest.raises(ValueError):
        trial.suggest_loguniform('y', 0., 1.)


def test_fixed_trial_suggest_discrete_uniform():
    # type: () -> None

    trial = FixedTrial({'x': 1.})
    assert trial.suggest_discrete_uniform('x', 0., 1., 0.1) == 1.

    with pytest.raises(ValueError):
        trial.suggest_discrete_uniform('y', 0., 1., 0.1)


def test_fixed_trial_suggest_int():
    # type: () -> None

    trial = FixedTrial({'x': 1})
    assert trial.suggest_int('x', 0, 10) == 1

    with pytest.raises(ValueError):
        trial.suggest_int('y', 0, 10)


def test_fixed_trial_suggest_categorical():
    # type: () -> None

    trial = FixedTrial({'x': 1})
    assert trial.suggest_categorical('x', [0, 1, 2, 3]) == 1

    with pytest.raises(ValueError):
        trial.suggest_categorical('y', [0, 1, 2, 3])


def test_fixed_trial_user_attrs():
    # type: () -> None

    trial = FixedTrial({'x': 1})
    trial.set_user_attr('data', 'MNIST')
    assert trial.user_attrs['data'] == 'MNIST'


def test_fixed_trial_system_attrs():
    # type: () -> None

    trial = FixedTrial({'x': 1})
    trial.set_system_attr('system_message', 'test')
    assert trial.system_attrs['system_message'] == 'test'


def test_fixed_trial_params():
    # type: () -> None

    params = {'x': 1}
    trial = FixedTrial(params)
    assert trial.params == params


def test_fixed_trial_report():
    # type: () -> None

    # FixedTrial ignores reported values.
    trial = FixedTrial({})
    trial.report(1.0, 1)
    trial.report(2.0)


def test_fixed_trial_should_prune():
    # type: () -> None

    # FixedTrial never prunes trials.
    assert FixedTrial({}).should_prune(1) is False
