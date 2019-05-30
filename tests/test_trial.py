import math
from mock import MagicMock
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
def test_suggest_low_equals_high(storage_init_func):
    # type: (typing.Callable[[], storages.BaseStorage]) -> None

    study = create_study(storage_init_func(), sampler=samplers.TPESampler(n_startup_trials=0))
    trial = Trial(study, study.storage.create_new_trial_id(study.study_id))

    # Parameter values are determined without suggestion when low == high.
    with patch.object(trial, '_suggest', wraps=trial._suggest) as mock_object:
        assert trial.suggest_uniform('a', 1., 1.) == 1.  # Suggesting a param.
        assert trial.suggest_uniform('a', 1., 1.) == 1.  # Suggesting the same param.
        assert mock_object.call_count == 0
        assert trial.suggest_loguniform('b', 1., 1.) == 1.  # Suggesting a param.
        assert trial.suggest_loguniform('b', 1., 1.) == 1.  # Suggesting the same param.
        assert mock_object.call_count == 0
        assert trial.suggest_discrete_uniform('c', 1., 1., 1.) == 1.  # Suggesting a param.
        assert trial.suggest_discrete_uniform('c', 1., 1., 1.) == 1.  # Suggesting the same param.
        assert mock_object.call_count == 0
        assert trial.suggest_int('d', 1, 1) == 1  # Suggesting a param.
        assert trial.suggest_int('d', 1, 1) == 1  # Suggesting the same param.
        assert mock_object.call_count == 0


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


@parametrize_storage
def test_distributions(storage_init_func):
    # type: (typing.Callable[[], storages.BaseStorage]) -> None

    def objective(trial):
        # type: (Trial) -> float

        trial.suggest_uniform('a', 0, 10)
        trial.suggest_loguniform('b', 0.1, 10)
        trial.suggest_discrete_uniform('c', 0, 10, 1)
        trial.suggest_int('d', 0, 10)
        trial.suggest_categorical('e', ['foo', 'bar', 'baz'])

        return 1.0

    study = create_study(storage_init_func())
    study.optimize(objective, n_trials=1)

    assert study.best_trial.distributions == {
        'a': distributions.UniformDistribution(low=0, high=10),
        'b': distributions.LogUniformDistribution(low=0.1, high=10),
        'c': distributions.DiscreteUniformDistribution(low=0, high=10, q=1),
        'd': distributions.IntUniformDistribution(low=0, high=10),
        'e': distributions.CategoricalDistribution(choices=('foo', 'bar', 'baz'))
    }


def test_trial_should_prune():
    # type: () -> None

    study_id = 1
    trial_id = 1

    study_mock = MagicMock()
    study_mock.study_id = study_id
    study_mock.storage.get_trial.return_value.\
        intermediate_values.keys.return_value = [1, 2, 3, 4, 5]
    study_mock.pruner.prune.return_value = True

    trial = Trial(study_mock, trial_id)  # type: ignore
    trial.should_prune()

    study_mock.storage.get_trial.assert_called_once_with(trial_id)
    study_mock.pruner.prune.assert_called_once_with(
        study_mock.storage, study_id, trial_id, 5,
    )


def test_fixed_trial_suggest_uniform():
    # type: () -> None

    trial = FixedTrial({'x': 1.})
    assert trial.suggest_uniform('x', -100., 100.) == 1.

    with pytest.raises(ValueError):
        trial.suggest_uniform('y', -100., 100.)


def test_fixed_trial_suggest_loguniform():
    # type: () -> None

    trial = FixedTrial({'x': 0.99})
    assert trial.suggest_loguniform('x', 0., 1.) == 0.99

    with pytest.raises(ValueError):
        trial.suggest_loguniform('y', 0., 1.)


def test_fixed_trial_suggest_discrete_uniform():
    # type: () -> None

    trial = FixedTrial({'x': 0.9})
    assert trial.suggest_discrete_uniform('x', 0., 1., 0.1) == 0.9

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

    # Integer categories.
    trial = FixedTrial({'x': 1})
    assert trial.suggest_categorical('x', [0, 1, 2, 3]) == 1

    with pytest.raises(ValueError):
        trial.suggest_categorical('y', [0, 1, 2, 3])

    # String categories.
    trial = FixedTrial({'x': 'baz'})
    assert trial.suggest_categorical('x', ['foo', 'bar', 'baz']) == 'baz'

    with pytest.raises(ValueError):
        trial.suggest_categorical('y', ['foo', 'bar', 'baz'])


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
    assert trial.params == {}

    assert trial.suggest_uniform('x', 0, 10) == 1
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
    assert FixedTrial({}).should_prune() is False
    assert FixedTrial({}).should_prune(1) is False
