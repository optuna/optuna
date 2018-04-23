from datetime import datetime
import pytest
from typing import Any  # NOQA
from typing import Callable  # NOQA
from typing import Dict  # NOQA

import pfnopt
from pfnopt.distributions import BaseDistribution  # NOQA
from pfnopt.distributions import CategoricalDistribution
from pfnopt.distributions import UniformDistribution
from pfnopt.storages import BaseStorage  # NOQA
from pfnopt.storages import InMemoryStorage
from pfnopt.storages import RDBStorage
from pfnopt.storages.base import SYSTEM_ATTRS_KEY


EXAMPLE_SYSTEM_ATTRS = {
    'dataset': 'MNIST',
    'none': None,
    'json_serializable': {'baseline_score': 0.001, 'tags': ['image', 'classification']},
}

EXAMPLE_USER_ATTRS = dict(EXAMPLE_SYSTEM_ATTRS, **{SYSTEM_ATTRS_KEY: {}})

EXAMPLE_DISTRIBUTIONS = {
    'x': UniformDistribution(low=1., high=2.),
    'y': CategoricalDistribution(choices=('Otemachi', 'Tokyo', 'Ginza'))
}  # type: Dict[str, BaseDistribution]

EXAMPLE_TRIALS = [
    pfnopt.trial.Trial(
        trial_id=-1,  # dummy id
        value=1.,
        state=pfnopt.trial.State.COMPLETE,
        user_attrs={SYSTEM_ATTRS_KEY: {}},
        params={'x': 0.5, 'y': 'Ginza'},
        intermediate_values={0: 2., 1: 3.},
        params_in_internal_repr={'x': .5, 'y': 2.},
        datetime_start=None,  # dummy
        datetime_complete=None  # dummy
    ),
    pfnopt.trial.Trial(
        trial_id=-1,  # dummy id
        value=2.,
        state=pfnopt.trial.State.RUNNING,
        user_attrs={
            SYSTEM_ATTRS_KEY: {'some_key': 'some_value'},
            'tags': ['video', 'classification'], 'dataset': 'YouTube-8M'},
        params={'x': 0.01, 'y': 'Otemachi'},
        intermediate_values={0: -2., 1: -3., 2: 100.},
        params_in_internal_repr={'x': .01, 'y': 0.},
        datetime_start=None,  # dummy
        datetime_complete=None  # dummy
    )
]


@pytest.mark.parametrize('storage_init_func', [
    InMemoryStorage,
    lambda: RDBStorage('sqlite:///:memory:')
])
def test_set_and_get_study_user_attrs(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()
    study_id = storage.create_new_study_id()

    def check_set_and_get(key, value):
        # type: (str, Any) -> None

        storage.set_study_user_attr(study_id, key, value)
        assert storage.get_study_user_attrs(study_id)[key] == value

    # Test setting value
    for key, value in EXAMPLE_USER_ATTRS.items():
        check_set_and_get(key, value)
    assert storage.get_study_user_attrs(study_id) == EXAMPLE_USER_ATTRS

    # Test overwriting value.
    check_set_and_get('dataset', 'ImageNet')


@pytest.mark.parametrize('storage_init_func', [
    InMemoryStorage,
    lambda: RDBStorage('sqlite:///:memory:')
])
def test_set_and_get_study_system_attrs(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()
    study_id = storage.create_new_study_id()

    def check_set_and_get(key, value):
        # type: (str, Any) -> None

        storage.set_study_system_attr(study_id, key, value)
        assert storage.get_study_user_attrs(study_id)[SYSTEM_ATTRS_KEY][key] == value
        assert storage.get_study_system_attr(study_id, key) == value

    # Test setting value.
    for key, value in EXAMPLE_SYSTEM_ATTRS.items():
        check_set_and_get(key, value)
    system_attrs = storage.get_study_user_attrs(study_id)[SYSTEM_ATTRS_KEY]
    assert system_attrs == EXAMPLE_SYSTEM_ATTRS

    # Test overwriting value.
    check_set_and_get('dataset', 'ImageNet')


@pytest.mark.parametrize('storage_init_func', [
    InMemoryStorage,
    lambda: RDBStorage('sqlite:///:memory:')
])
def test_set_trial_user_attrs(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()
    trial_id_1 = storage.create_new_trial_id(storage.create_new_study_id())

    def check_set_and_get(trial_id, key, value):
        # type: (int, str, Any) -> None

        storage.set_trial_user_attr(trial_id, key, value)
        assert storage.get_trial(trial_id).user_attrs[key] == value

    # Test setting value.
    for key, value in EXAMPLE_USER_ATTRS.items():
        check_set_and_get(trial_id_1, key, value)
    assert storage.get_trial(trial_id_1).user_attrs == EXAMPLE_USER_ATTRS

    # Test overwriting value.
    check_set_and_get(trial_id_1, 'dataset', 'ImageNet')

    # Test another trial.
    trial_id_2 = storage.create_new_trial_id(storage.create_new_study_id())
    check_set_and_get(trial_id_2, 'baseline_score', 0.001)
    assert len(storage.get_trial(trial_id_2).user_attrs) == 2
    assert storage.get_trial(trial_id_2).user_attrs['baseline_score'] == 0.001


@pytest.mark.parametrize('storage_init_func', [
    InMemoryStorage,
    lambda: RDBStorage('sqlite:///:memory:')
])
def test_set_and_get_tiral_system_attr(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()
    trial_id_1 = storage.create_new_trial_id(storage.create_new_study_id())

    def check_set_and_get(trial_id, key, value):
        # type: (int, str, Any) -> None

        storage.set_trial_system_attr(trial_id, key, value)
        assert storage.get_trial(trial_id).user_attrs[SYSTEM_ATTRS_KEY][key] == value
        assert storage.get_trial_system_attr(trial_id, key) == value

    # Test setting value.
    for key, value in EXAMPLE_SYSTEM_ATTRS.items():
        check_set_and_get(trial_id_1, key, value)
    system_attrs = storage.get_trial(trial_id_1).user_attrs[SYSTEM_ATTRS_KEY]
    assert system_attrs == EXAMPLE_SYSTEM_ATTRS

    # Test overwriting value.
    check_set_and_get(trial_id_1, 'dataset', 'ImageNet')

    # Test another trial.
    trial_id_2 = storage.create_new_trial_id(storage.create_new_study_id())
    check_set_and_get(trial_id_2, 'baseline_score', 0.001)
    system_attrs = storage.get_trial(trial_id_2).user_attrs[SYSTEM_ATTRS_KEY]
    assert system_attrs == {'baseline_score': 0.001}


@pytest.mark.parametrize('storage_init_func', [
    InMemoryStorage,
    lambda: RDBStorage('sqlite:///:memory:')
])
def test_get_trial(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()
    study_id = storage.create_new_study_id()

    for example_trial in EXAMPLE_TRIALS:
        datetime_before = datetime.now()

        trial_id = _create_new_trial_with_example_trial(
            storage, study_id, EXAMPLE_DISTRIBUTIONS, example_trial)

        datetime_after = datetime.now()

        trial = storage.get_trial(trial_id)
        _check_example_trial_static_attributes(trial, example_trial)
        if trial.state.is_finished():
            assert datetime_before < trial.datetime_start < datetime_after
            assert datetime_before < trial.datetime_complete < datetime_after
        else:
            assert datetime_before < trial.datetime_start < datetime_after
            assert trial.datetime_complete is None


def _create_new_trial_with_example_trial(storage, study_id, distributions, example_trial):
    # type: (BaseStorage, int, Dict[str, BaseDistribution], pfnopt.trial.Trial) -> int

    trial_id = storage.create_new_trial_id(study_id)

    storage.set_trial_value(trial_id, example_trial.value)
    storage.set_trial_state(trial_id, example_trial.state)
    _set_distributions(storage, trial_id, distributions)

    for name, ex_repr in example_trial.params.items():
        storage.set_trial_param(trial_id, name, distributions[name].to_internal_repr(ex_repr))

    for step, value in example_trial.intermediate_values.items():
        storage.set_trial_intermediate_value(trial_id, step, value)

    for key, value in example_trial.user_attrs.items():
        storage.set_trial_user_attr(trial_id, key, value)

    return trial_id


def _set_distributions(storage, trial_id, distributions):
    # type: (BaseStorage, int, Dict[str, BaseDistribution]) -> None

    for k, v in distributions.items():
        storage.set_trial_param_distribution(trial_id, k, v)


def _check_example_trial_static_attributes(trial_1, trial_2):
    # type: (pfnopt.trial.Trial, pfnopt.trial.Trial) -> None

    trial_1 = trial_1._replace(trial_id=-1, datetime_start=None, datetime_complete=None)
    trial_2 = trial_2._replace(trial_id=-1, datetime_start=None, datetime_complete=None)
    assert trial_1 == trial_2
