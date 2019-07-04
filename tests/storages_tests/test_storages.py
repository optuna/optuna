from datetime import datetime
import math
from mock import patch
import pytest

import optuna
from optuna.distributions import BaseDistribution  # NOQA
from optuna.distributions import CategoricalDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.storages.base import DEFAULT_STUDY_NAME_PREFIX
from optuna.storages import BaseStorage  # NOQA
from optuna.storages import InMemoryStorage
from optuna.storages import RDBStorage
from optuna.structs import FrozenTrial
from optuna.structs import StudyDirection
from optuna.structs import TrialState
from optuna.testing.storage import StorageSupplier
from optuna import types

if types.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Callable  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA

# TODO(Yanase): Remove _number from system_attrs after adding TrialModel.number.
EXAMPLE_ATTRS = {
    'dataset': 'MNIST',
    'none': None,
    'json_serializable': {
        'baseline_score': 0.001,
        'tags': ['image', 'classification']
    },
    '_number': 0,
}

EXAMPLE_DISTRIBUTIONS = {
    'x': UniformDistribution(low=1., high=2.),
    'y': CategoricalDistribution(choices=('Otemachi', 'Tokyo', 'Ginza'))
}  # type: Dict[str, BaseDistribution]

# TODO(Yanase): Remove _number from system_attrs after adding TrialModel.number.
EXAMPLE_TRIALS = [
    FrozenTrial(
        number=0,  # dummy
        value=1.,
        state=TrialState.COMPLETE,
        user_attrs={},
        system_attrs={'_number': 0},
        params={
            'x': 0.5,
            'y': 'Ginza'
        },
        distributions=EXAMPLE_DISTRIBUTIONS,
        intermediate_values={
            0: 2.,
            1: 3.
        },
        params_in_internal_repr={
            'x': .5,
            'y': 2.
        },
        datetime_start=None,  # dummy
        datetime_complete=None,  # dummy
        trial_id=-1,  # dummy id
    ),
    FrozenTrial(
        number=0,  # dummy
        value=2.,
        state=TrialState.RUNNING,
        user_attrs={
            'tags': ['video', 'classification'],
            'dataset': 'YouTube-8M'
        },
        system_attrs={'some_key': 'some_value', '_number': 0},
        params={
            'x': 0.01,
            'y': 'Otemachi'
        },
        distributions=EXAMPLE_DISTRIBUTIONS,
        intermediate_values={
            0: -2.,
            1: -3.,
            2: 100.
        },
        params_in_internal_repr={
            'x': .01,
            'y': 0.
        },
        datetime_start=None,  # dummy
        datetime_complete=None,  # dummy
        trial_id=-1,  # dummy id
    )
]

STORAGE_MODES = [
    'none',  # We give `None` to storage argument, so InMemoryStorage is used.
    'new',  # We always create a new sqlite DB file for each experiment.
    'common',  # We use a sqlite DB file for the whole experiments.
]

CACHE_MODES = [True, False]

# TODO(Yanase): Replace @parametrize_storage with StorageSupplier.
parametrize_storage = pytest.mark.parametrize(
    'storage_init_func', [InMemoryStorage, lambda: RDBStorage('sqlite:///:memory:')])


def setup_module():
    # type: () -> None

    StorageSupplier.setup_common_tempfile()


def teardown_module():
    # type: () -> None

    StorageSupplier.teardown_common_tempfile()


@parametrize_storage
def test_create_new_study_id(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()
    study_id = storage.create_new_study_id()

    summaries = storage.get_all_study_summaries()
    assert len(summaries) == 1
    assert summaries[0].study_id == study_id
    assert summaries[0].study_name.startswith(DEFAULT_STUDY_NAME_PREFIX)


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_create_new_study_id_with_name(storage_mode, cache_mode):
    # type: (str, bool) -> None

    with StorageSupplier(storage_mode, cache_mode) as storage:

        # Generate unique study_name from the current function name and storage_mode.
        function_name = test_create_new_study_id_with_name.__name__
        study_name = function_name + '/' + storage_mode + '/' + str(cache_mode)
        storage = optuna.storages.get_storage(storage)
        study_id = storage.create_new_study_id(study_name)

        assert study_name == storage.get_study_name_from_id(study_id)


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_get_study_id_from_name_and_get_study_name_from_id(storage_mode, cache_mode):
    # type: (str, bool) -> None

    with StorageSupplier(storage_mode, cache_mode) as storage:

        # Generate unique study_name from the current function name and storage_mode.
        function_name = test_get_study_id_from_name_and_get_study_name_from_id.__name__
        study_name = function_name + '/' + storage_mode + '/' + str(cache_mode)
        storage = optuna.storages.get_storage(storage)
        study = optuna.create_study(storage=storage, study_name=study_name)

        # Test existing study.
        assert storage.get_study_name_from_id(study.study_id) == study_name
        assert storage.get_study_id_from_name(study_name) == study.study_id

        # Test not existing study.
        with pytest.raises(ValueError):
            storage.get_study_id_from_name('dummy-name')

        with pytest.raises(ValueError):
            storage.get_study_name_from_id(-1)


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_get_study_id_from_trial_id(storage_mode, cache_mode):
    # type: (str, bool) -> None

    with StorageSupplier(storage_mode, cache_mode) as storage:

        # Generate unique study_name from the current function name and storage_mode.
        storage = optuna.storages.get_storage(storage)

        # Check if trial_number starts from 0.
        study_id = storage.create_new_study_id()

        trial_id = storage.create_new_trial_id(study_id)
        assert storage.get_study_id_from_trial_id(trial_id) == study_id


@parametrize_storage
def test_set_and_get_study_direction(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()
    study_id = storage.create_new_study_id()

    def check_set_and_get(direction):
        # type: (StudyDirection) -> None

        storage.set_study_direction(study_id, direction)
        assert storage.get_study_direction(study_id) == direction

    assert storage.get_study_direction(study_id) == StudyDirection.NOT_SET

    # Test setting value.
    check_set_and_get(StudyDirection.MINIMIZE)

    # Test overwriting value.
    with pytest.raises(ValueError):
        storage.set_study_direction(study_id, StudyDirection.MAXIMIZE)


@parametrize_storage
def test_set_and_get_study_user_attrs(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()
    study_id = storage.create_new_study_id()

    def check_set_and_get(key, value):
        # type: (str, Any) -> None

        storage.set_study_user_attr(study_id, key, value)
        assert storage.get_study_user_attrs(study_id)[key] == value

    # Test setting value.
    for key, value in EXAMPLE_ATTRS.items():
        check_set_and_get(key, value)
    assert storage.get_study_user_attrs(study_id) == EXAMPLE_ATTRS

    # Test overwriting value.
    check_set_and_get('dataset', 'ImageNet')


@parametrize_storage
def test_set_and_get_study_system_attrs(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()
    study_id = storage.create_new_study_id()

    def check_set_and_get(key, value):
        # type: (str, Any) -> None

        storage.set_study_system_attr(study_id, key, value)
        assert storage.get_study_system_attrs(study_id)[key] == value

    # Test setting value.
    for key, value in EXAMPLE_ATTRS.items():
        check_set_and_get(key, value)

    # Test overwriting value.
    check_set_and_get('dataset', 'ImageNet')


@parametrize_storage
def test_create_new_trial_id(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()

    study_id = storage.create_new_study_id()
    trial_id = storage.create_new_trial_id(study_id)

    trials = storage.get_all_trials(study_id)
    assert len(trials) == 1
    assert trials[0].trial_id == trial_id
    assert trials[0].number == 0
    assert trials[0].state == TrialState.RUNNING
    assert trials[0].user_attrs == {}

    # TODO(Yanase): Remove number from system_attrs after adding TrialModel.number.
    assert trials[0].system_attrs == {'_number': 0}


@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_get_trial_number_from_id(storage_mode, cache_mode):
    # type: (str, bool) -> None

    with StorageSupplier(storage_mode, cache_mode) as storage:
        storage = optuna.storages.get_storage(storage)

        # Check if trial_number starts from 0.
        study_id = storage.create_new_study_id()

        trial_id = storage.create_new_trial_id(study_id)
        assert storage.get_trial_number_from_id(trial_id) == 0

        trial_id = storage.create_new_trial_id(study_id)
        assert storage.get_trial_number_from_id(trial_id) == 1


# TODO(Yanase): Remove the following test case after TrialModel.number is added.
@pytest.mark.parametrize('storage_mode', STORAGE_MODES)
@pytest.mark.parametrize('cache_mode', CACHE_MODES)
def test_get_trial_number_from_id_with_empty_system_attrs(storage_mode, cache_mode):
    # type: (str, bool) -> None

    with StorageSupplier(storage_mode, cache_mode) as storage:
        storage = optuna.storages.get_storage(storage)
        study_id = storage.create_new_study_id()
        with patch.object(storage, 'get_trial_system_attrs', return_value=dict()) as _mock_attrs:
            trial_id = storage.create_new_trial_id(study_id)
            assert storage.get_trial_number_from_id(trial_id) == 0

            trial_id = storage.create_new_trial_id(study_id)
            assert storage.get_trial_number_from_id(trial_id) == 1

            if storage_mode == 'none':
                return
            assert _mock_attrs.call_count == 2


@parametrize_storage
def test_set_trial_state(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()

    trial_id_1 = storage.create_new_trial_id(storage.create_new_study_id())
    trial_id_2 = storage.create_new_trial_id(storage.create_new_study_id())

    storage.set_trial_state(trial_id_1, TrialState.RUNNING)
    assert storage.get_trial(trial_id_1).state == TrialState.RUNNING
    assert storage.get_trial(trial_id_1).datetime_complete is None

    storage.set_trial_state(trial_id_2, TrialState.COMPLETE)
    assert storage.get_trial(trial_id_2).state == TrialState.COMPLETE
    assert storage.get_trial(trial_id_2).datetime_complete is not None

    # Test overwriting value.
    storage.set_trial_state(trial_id_1, TrialState.PRUNED)
    assert storage.get_trial(trial_id_1).state == TrialState.PRUNED
    assert storage.get_trial(trial_id_1).datetime_complete is not None


@parametrize_storage
def test_set_and_get_trial_param(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()

    # Setup test across multiple studies and trials.
    study_id = storage.create_new_study_id()
    trial_id_1 = storage.create_new_trial_id(study_id)
    trial_id_2 = storage.create_new_trial_id(study_id)
    trial_id_3 = storage.create_new_trial_id(storage.create_new_study_id())

    # Setup Distributions.
    distribution_x = UniformDistribution(low=1.0, high=2.0)
    distribution_y_1 = CategoricalDistribution(choices=('Shibuya', 'Ebisu', 'Meguro'))
    distribution_y_2 = CategoricalDistribution(choices=('Shibuya', 'Shinsen'))
    distribution_z = LogUniformDistribution(low=1.0, high=100.0)

    # Test trial_1: setting new params.
    assert storage.set_trial_param(trial_id_1, 'x', 0.5, distribution_x)
    assert storage.set_trial_param(trial_id_1, 'y', 2, distribution_y_1)

    # Test trial_1: getting params.
    assert storage.get_trial_param(trial_id_1, 'x') == 0.5
    assert storage.get_trial_param(trial_id_1, 'y') == 2
    # Test trial_1: checking all params and external repr.
    assert storage.get_trial(trial_id_1).params == {'x': 0.5, 'y': 'Meguro'}
    # Test trial_1: setting existing name.
    assert not storage.set_trial_param(trial_id_1, 'x', 0.6, distribution_x)

    # Setup trial_2: setting new params (to the same study as trial_1).
    assert storage.set_trial_param(trial_id_2, 'x', 0.3, distribution_x)
    assert storage.set_trial_param(trial_id_2, 'z', 0.1, distribution_z)

    # Test trial_2: getting params.
    assert storage.get_trial_param(trial_id_2, 'x') == 0.3
    assert storage.get_trial_param(trial_id_2, 'z') == 0.1

    # Test trial_2: checking all params and external repr.
    assert storage.get_trial(trial_id_2).params == {'x': 0.3, 'z': 0.1}
    # Test trial_2: setting different distribution.
    with pytest.raises(ValueError):
        storage.set_trial_param(trial_id_2, 'x', 0.5, distribution_z)
    # Test trial_2: setting CategoricalDistribution in different order.
    with pytest.raises(ValueError):
        storage.set_trial_param(trial_id_2, 'y', 2,
                                CategoricalDistribution(choices=('Meguro', 'Shibuya', 'Ebisu')))

    # Setup trial_3: setting new params (to different study from trial_1).
    if isinstance(storage, InMemoryStorage):
        with pytest.raises(ValueError):
            # InMemoryStorage shares the same study if create_new_study_id is additionally invoked.
            # Thus, the following line should fail due to distribution incompatibility.
            storage.set_trial_param(trial_id_3, 'y', 1, distribution_y_2)
    else:
        assert storage.set_trial_param(trial_id_3, 'y', 1, distribution_y_2)
        assert storage.get_trial_param(trial_id_3, 'y') == 1
        assert storage.get_trial(trial_id_3).params == {'y': 'Shinsen'}


@parametrize_storage
def test_set_trial_value(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()

    # Setup test across multiple studies and trials.
    study_id = storage.create_new_study_id()
    trial_id_1 = storage.create_new_trial_id(study_id)
    trial_id_2 = storage.create_new_trial_id(study_id)
    trial_id_3 = storage.create_new_trial_id(storage.create_new_study_id())

    # Test setting new value.
    storage.set_trial_value(trial_id_1, 0.5)
    storage.set_trial_value(trial_id_3, float('inf'))

    assert storage.get_trial(trial_id_1).value == 0.5
    assert storage.get_trial(trial_id_2).value is None
    assert storage.get_trial(trial_id_3).value == float('inf')


@parametrize_storage
def test_set_trial_intermediate_value(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()

    # Setup test across multiple studies and trials.
    study_id = storage.create_new_study_id()
    trial_id_1 = storage.create_new_trial_id(study_id)
    trial_id_2 = storage.create_new_trial_id(study_id)
    trial_id_3 = storage.create_new_trial_id(storage.create_new_study_id())

    # Test setting new values.
    assert storage.set_trial_intermediate_value(trial_id_1, 0, 0.3)
    assert storage.set_trial_intermediate_value(trial_id_1, 2, 0.4)
    assert storage.set_trial_intermediate_value(trial_id_3, 0, 0.1)
    assert storage.set_trial_intermediate_value(trial_id_3, 1, 0.4)
    assert storage.set_trial_intermediate_value(trial_id_3, 2, 0.5)

    assert storage.get_trial(trial_id_1).intermediate_values == {0: 0.3, 2: 0.4}
    assert storage.get_trial(trial_id_2).intermediate_values == {}
    assert storage.get_trial(trial_id_3).intermediate_values == {0: 0.1, 1: 0.4, 2: 0.5}

    # Test setting existing step.
    assert not storage.set_trial_intermediate_value(trial_id_1, 0, 0.3)


@parametrize_storage
def test_set_trial_user_attr(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()
    trial_id_1 = storage.create_new_trial_id(storage.create_new_study_id())

    def check_set_and_get(trial_id, key, value):
        # type: (int, str, Any) -> None

        storage.set_trial_user_attr(trial_id, key, value)
        assert storage.get_trial(trial_id).user_attrs[key] == value

    # Test setting value.
    for key, value in EXAMPLE_ATTRS.items():
        check_set_and_get(trial_id_1, key, value)
    assert storage.get_trial(trial_id_1).user_attrs == EXAMPLE_ATTRS

    # Test overwriting value.
    check_set_and_get(trial_id_1, 'dataset', 'ImageNet')

    # Test another trial.
    trial_id_2 = storage.create_new_trial_id(storage.create_new_study_id())
    check_set_and_get(trial_id_2, 'baseline_score', 0.001)
    assert len(storage.get_trial(trial_id_2).user_attrs) == 1
    assert storage.get_trial(trial_id_2).user_attrs['baseline_score'] == 0.001


@parametrize_storage
def test_set_and_get_tiral_system_attr(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()
    study_id = storage.create_new_study_id()
    trial_id_1 = storage.create_new_trial_id(study_id)

    def check_set_and_get(trial_id, key, value):
        # type: (int, str, Any) -> None

        storage.set_trial_system_attr(trial_id, key, value)
        assert storage.get_trial_system_attrs(trial_id)[key] == value

    # Test setting value.
    for key, value in EXAMPLE_ATTRS.items():
        check_set_and_get(trial_id_1, key, value)
    system_attrs = storage.get_trial(trial_id_1).system_attrs
    assert system_attrs == EXAMPLE_ATTRS

    # Test overwriting value.
    check_set_and_get(trial_id_1, 'dataset', 'ImageNet')

    # Test another trial.
    trial_id_2 = storage.create_new_trial_id(study_id)
    check_set_and_get(trial_id_2, 'baseline_score', 0.001)
    system_attrs = storage.get_trial(trial_id_2).system_attrs
    # TODO(Yanase): Remove number from system_attrs after adding TrialModel.number.
    assert system_attrs == {'baseline_score': 0.001, '_number': 1}


@parametrize_storage
def test_get_all_study_summaries(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()
    study_id = storage.create_new_study_id()

    storage.set_study_direction(study_id, StudyDirection.MINIMIZE)

    datetime_1 = datetime.now()

    # Set up trial 1.
    _create_new_trial_with_example_trial(storage, study_id, EXAMPLE_DISTRIBUTIONS,
                                         EXAMPLE_TRIALS[0])

    datetime_2 = datetime.now()

    # Set up trial 2.
    trial_id_2 = storage.create_new_trial_id(study_id)
    storage.set_trial_value(trial_id_2, 2.0)

    for key, value in EXAMPLE_ATTRS.items():
        storage.set_study_user_attr(study_id, key, value)

    summaries = storage.get_all_study_summaries()

    assert len(summaries) == 1
    assert summaries[0].study_id == study_id
    assert summaries[0].study_name == storage.get_study_name_from_id(study_id)
    assert summaries[0].direction == StudyDirection.MINIMIZE
    assert summaries[0].user_attrs == EXAMPLE_ATTRS
    assert summaries[0].n_trials == 2
    assert summaries[0].datetime_start is not None
    assert datetime_1 < summaries[0].datetime_start < datetime_2
    _check_example_trial_static_attributes(summaries[0].best_trial, EXAMPLE_TRIALS[0])


@parametrize_storage
def test_get_trial(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()
    study_id = storage.create_new_study_id()

    for example_trial in EXAMPLE_TRIALS:
        datetime_before = datetime.now()

        trial_id = _create_new_trial_with_example_trial(storage, study_id, EXAMPLE_DISTRIBUTIONS,
                                                        example_trial)

        datetime_after = datetime.now()

        trial = storage.get_trial(trial_id)
        _check_example_trial_static_attributes(trial, example_trial)
        if trial.state.is_finished():
            assert trial.datetime_start is not None
            assert trial.datetime_complete is not None
            assert datetime_before < trial.datetime_start < datetime_after
            assert datetime_before < trial.datetime_complete < datetime_after
        else:
            assert trial.datetime_start is not None
            assert trial.datetime_complete is None
            assert datetime_before < trial.datetime_start < datetime_after


@parametrize_storage
def test_get_all_trials(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()
    study_id_1 = storage.create_new_study_id()
    study_id_2 = storage.create_new_study_id()

    datetime_before = datetime.now()

    _create_new_trial_with_example_trial(storage, study_id_1, EXAMPLE_DISTRIBUTIONS,
                                         EXAMPLE_TRIALS[0])
    _create_new_trial_with_example_trial(storage, study_id_1, EXAMPLE_DISTRIBUTIONS,
                                         EXAMPLE_TRIALS[1])
    _create_new_trial_with_example_trial(storage, study_id_2, EXAMPLE_DISTRIBUTIONS,
                                         EXAMPLE_TRIALS[0])

    datetime_after = datetime.now()

    # Test getting multiple trials.
    trials = sorted(storage.get_all_trials(study_id_1), key=lambda x: x.trial_id)
    _check_example_trial_static_attributes(trials[0], EXAMPLE_TRIALS[0])
    _check_example_trial_static_attributes(trials[1], EXAMPLE_TRIALS[1])
    for t in trials:
        assert t.datetime_start is not None
        assert datetime_before < t.datetime_start < datetime_after
        if t.state.is_finished():
            assert t.datetime_complete is not None
            assert datetime_before < t.datetime_complete < datetime_after
        else:
            assert t.datetime_complete is None

    # Test getting trials per study.
    trials = sorted(storage.get_all_trials(study_id_2), key=lambda x: x.trial_id)
    _check_example_trial_static_attributes(trials[0], EXAMPLE_TRIALS[0])


@parametrize_storage
def test_get_n_trials(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    storage = storage_init_func()
    study_id = storage.create_new_study_id()

    _create_new_trial_with_example_trial(storage, study_id, EXAMPLE_DISTRIBUTIONS,
                                         EXAMPLE_TRIALS[0])
    _create_new_trial_with_example_trial(storage, study_id, EXAMPLE_DISTRIBUTIONS,
                                         EXAMPLE_TRIALS[1])

    assert 2 == storage.get_n_trials(study_id)
    assert 1 == storage.get_n_trials(study_id, TrialState.COMPLETE)


@parametrize_storage
@pytest.mark.parametrize('direction_expected', [(StudyDirection.MINIMIZE, 0.1),
                                                (StudyDirection.MAXIMIZE, 0.2)])
def test_get_best_intermediate_result_over_steps(storage_init_func, direction_expected):
    # type: (Callable[[], BaseStorage], Tuple[StudyDirection, float]) -> None

    direction, expected = direction_expected

    storage = storage_init_func()
    study_id = storage.create_new_study_id()
    storage.set_study_direction(study_id, direction)

    # FrozenTrial.intermediate_values has no elements.
    trial_id_empty = storage.create_new_trial_id(study_id)
    with pytest.raises(ValueError):
        storage.get_best_intermediate_result_over_steps(trial_id_empty)

    # Input value has no NaNs but float values.
    trial_id_float = storage.create_new_trial_id(study_id)
    storage.set_trial_intermediate_value(trial_id_float, 0, 0.1)
    storage.set_trial_intermediate_value(trial_id_float, 1, 0.2)
    assert expected == storage.get_best_intermediate_result_over_steps(trial_id_float)

    # Input value has a float value and a NaN.
    trial_id_float_nan = storage.create_new_trial_id(study_id)
    storage.set_trial_intermediate_value(trial_id_float_nan, 0, 0.3)
    storage.set_trial_intermediate_value(trial_id_float_nan, 1, float('nan'))
    assert 0.3 == storage.get_best_intermediate_result_over_steps(trial_id_float_nan)

    # Input value has a NaN only.
    trial_id_nan = storage.create_new_trial_id(study_id)
    storage.set_trial_intermediate_value(trial_id_nan, 0, float('nan'))
    assert math.isnan(storage.get_best_intermediate_result_over_steps(trial_id_nan))


@parametrize_storage
def test_get_percentile_intermediate_result_over_trials(storage_init_func):
    # type: (Callable[[], BaseStorage]) -> None

    def setup_study(trial_num, intermediate_values):
        # type: (int, List[List[float]]) -> Tuple[int, BaseStorage]

        storage = storage_init_func()
        study_id = storage.create_new_study_id()
        trial_ids = [storage.create_new_trial_id(study_id) for _ in range(trial_num)]

        for step, values in enumerate(intermediate_values):
            # Study does not have any trials.
            with pytest.raises(ValueError):
                storage.get_percentile_intermediate_result_over_trials(study_id, step, 25)

            for i in range(trial_num):
                trial_id = trial_ids[i]
                value = values[i]
                storage.set_trial_intermediate_value(trial_id, step, value)

        # Set trial states complete because this method ignores incomplete trials.
        for trial_id in trial_ids:
            storage.set_trial_state(trial_id, TrialState.COMPLETE)

        return study_id, storage

    # Input value has no NaNs but float values (step=0).
    intermediate_values = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    study_id, storage = setup_study(9, intermediate_values)
    assert 0.3 == storage.get_percentile_intermediate_result_over_trials(study_id, 0, 25.0)

    # Input value has a float value and NaNs (step=1).
    intermediate_values.append([
        0.1, 0.2, 0.3, 0.4, 0.5,
        float('nan'), float('nan'), float('nan'), float('nan')])
    study_id, storage = setup_study(9, intermediate_values)
    assert 0.2 == storage.get_percentile_intermediate_result_over_trials(study_id, 1, 25.0)

    # Input value has NaNs only (step=2).
    intermediate_values.append([
        float('nan'), float('nan'), float('nan'), float('nan'), float('nan'),
        float('nan'), float('nan'), float('nan'), float('nan')])
    study_id, storage = setup_study(9, intermediate_values)
    assert math.isnan(storage.get_percentile_intermediate_result_over_trials(study_id, 2, 75))


def _create_new_trial_with_example_trial(storage, study_id, distributions, example_trial):
    # type: (BaseStorage, int, Dict[str, BaseDistribution], FrozenTrial) -> int

    trial_id = storage.create_new_trial_id(study_id)

    if example_trial.value is not None:
        storage.set_trial_value(trial_id, example_trial.value)

    for name, param_external in example_trial.params.items():
        param_internal = distributions[name].to_internal_repr(param_external)
        distribution = distributions[name]
        storage.set_trial_param(trial_id, name, param_internal, distribution)

    for step, value in example_trial.intermediate_values.items():
        storage.set_trial_intermediate_value(trial_id, step, value)

    for key, value in example_trial.user_attrs.items():
        storage.set_trial_user_attr(trial_id, key, value)

    for key, value in example_trial.system_attrs.items():
        storage.set_trial_system_attr(trial_id, key, value)

    storage.set_trial_state(trial_id, example_trial.state)

    return trial_id


def _check_example_trial_static_attributes(trial_1, trial_2):
    # type: (Optional[FrozenTrial], Optional[FrozenTrial]) -> None

    assert trial_1 is not None
    assert trial_2 is not None

    trial_1 = trial_1._replace(trial_id=-1, number=0, datetime_start=None, datetime_complete=None)
    trial_2 = trial_2._replace(trial_id=-1, number=0, datetime_start=None, datetime_complete=None)

    assert trial_1 == trial_2
