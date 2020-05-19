import copy
from datetime import datetime
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from unittest.mock import patch

import pytest

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.storages.base import DEFAULT_STUDY_NAME_PREFIX
from optuna.storages import BaseStorage
from optuna.storages.cached_storage import _CachedStorage
from optuna.storages import InMemoryStorage
from optuna.storages import RDBStorage
from optuna.storages import RedisStorage
from optuna.study import StudyDirection
from optuna.study import StudySummary
from optuna.testing.storage import StorageSupplier
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


ALL_STATES = list(TrialState)

EXAMPLE_ATTRS = {
    "dataset": "MNIST",
    "none": None,
    "json_serializable": {"baseline_score": 0.001, "tags": ["image", "classification"]},
}

STORAGE_MODES = [
    "inmemory",
    "sqlite",
    "redis",
    "cache",
]


def test_get_storage() -> None:

    assert isinstance(optuna.storages.get_storage(None), InMemoryStorage)
    assert isinstance(optuna.storages.get_storage("sqlite:///:memory:"), _CachedStorage)
    assert isinstance(
        optuna.storages.get_storage("redis://test_user:passwd@localhost:6379/0"), RedisStorage
    )


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_create_new_study(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:

        study_id = storage.create_new_study()

        summaries = storage.get_all_study_summaries()
        assert len(summaries) == 1
        assert summaries[0]._study_id == study_id
        assert summaries[0].study_name.startswith(DEFAULT_STUDY_NAME_PREFIX)

        study_id2 = storage.create_new_study()
        # Study id must be unique.
        assert study_id != study_id2
        summaries = storage.get_all_study_summaries()
        assert len(summaries) == 2
        assert {s._study_id for s in summaries} == {study_id, study_id2}
        assert all(s.study_name.startswith(DEFAULT_STUDY_NAME_PREFIX) for s in summaries)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_create_new_study_unique_id(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:

        study_id = storage.create_new_study()
        study_id2 = storage.create_new_study()
        storage.delete_study(study_id2)
        study_id3 = storage.create_new_study()

        # Study id must not be reused after deletion.
        if not (isinstance(storage, RDBStorage) or isinstance(storage, _CachedStorage)):
            # TODO(ytsmiling) Fix RDBStorage so that it does not reuse study_id.
            assert len({study_id, study_id2, study_id3}) == 3
        summaries = storage.get_all_study_summaries()
        assert {s._study_id for s in summaries} == {study_id, study_id3}


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_create_new_study_with_name(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:

        # Generate unique study_name from the current function name and storage_mode.
        function_name = test_create_new_study_with_name.__name__
        study_name = function_name + "/" + storage_mode
        study_id = storage.create_new_study(study_name)

        assert study_name == storage.get_study_name_from_id(study_id)

        with pytest.raises(optuna.exceptions.DuplicatedStudyError):
            storage.create_new_study(study_name)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_delete_study(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:

        study_id = storage.create_new_study()
        storage.create_new_trial(study_id)
        trials = storage.get_all_trials(study_id)
        assert len(trials) == 1

        with pytest.raises(KeyError):
            # Deletion of non-existent study.
            storage.delete_study(study_id + 1)

        storage.delete_study(study_id)
        study_id = storage.create_new_study()
        trials = storage.get_all_trials(study_id)
        assert len(trials) == 0

        storage.delete_study(study_id)
        with pytest.raises(KeyError):
            # Double free.
            storage.delete_study(study_id)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_delete_study_after_create_multiple_studies(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study_id1 = storage.create_new_study()
        study_id2 = storage.create_new_study()
        study_id3 = storage.create_new_study()

        storage.delete_study(study_id2)

        studies = {s._study_id: s for s in storage.get_all_study_summaries()}
        assert study_id1 in studies
        assert study_id2 not in studies
        assert study_id3 in studies


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_study_id_from_name_and_get_study_name_from_id(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:

        # Generate unique study_name from the current function name and storage_mode.
        function_name = test_get_study_id_from_name_and_get_study_name_from_id.__name__
        study_name = function_name + "/" + storage_mode
        storage = optuna.storages.get_storage(storage)
        study_id = storage.create_new_study(study_name=study_name)

        # Test existing study.
        assert storage.get_study_name_from_id(study_id) == study_name
        assert storage.get_study_id_from_name(study_name) == study_id

        # Test not existing study.
        with pytest.raises(KeyError):
            storage.get_study_id_from_name("dummy-name")

        with pytest.raises(KeyError):
            storage.get_study_name_from_id(study_id + 1)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_study_id_from_trial_id(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:

        # Generate unique study_name from the current function name and storage_mode.
        storage = optuna.storages.get_storage(storage)

        # Check if trial_number starts from 0.
        study_id = storage.create_new_study()

        trial_id = storage.create_new_trial(study_id)
        assert storage.get_study_id_from_trial_id(trial_id) == study_id


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_set_and_get_study_direction(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:

        for target, opposite in [
            (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE),
            (StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE),
        ]:

            study_id = storage.create_new_study()

            def check_set_and_get(direction: StudyDirection) -> None:
                storage.set_study_direction(study_id, direction)
                assert storage.get_study_direction(study_id) == direction

            assert storage.get_study_direction(study_id) == StudyDirection.NOT_SET

            # Test setting value.
            check_set_and_get(target)

            # Test overwriting value to the same direction.
            storage.set_study_direction(study_id, target)

            # Test overwriting value to the opposite direction.
            with pytest.raises(ValueError):
                storage.set_study_direction(study_id, opposite)

            # Test overwriting value to the not set.
            with pytest.raises(ValueError):
                storage.set_study_direction(study_id, StudyDirection.NOT_SET)

            # Test non-existent study.
            with pytest.raises(KeyError):
                storage.set_study_direction(study_id + 1, opposite)

            # Test non-existent study is checked before directions.
            with pytest.raises(KeyError):
                storage.set_study_direction(study_id + 1, StudyDirection.NOT_SET)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_set_and_get_study_user_attrs(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study()

        def check_set_and_get(key: str, value: Any) -> None:

            storage.set_study_user_attr(study_id, key, value)
            assert storage.get_study_user_attrs(study_id)[key] == value

        # Test setting value.
        for key, value in EXAMPLE_ATTRS.items():
            check_set_and_get(key, value)
        assert storage.get_study_user_attrs(study_id) == EXAMPLE_ATTRS

        # Test overwriting value.
        check_set_and_get("dataset", "ImageNet")

        # Non-existent study id or key.
        non_existent_study_id = study_id + 1
        with pytest.raises(KeyError):
            storage.set_study_user_attr(non_existent_study_id, "key", "value")
        with pytest.raises(KeyError):
            storage.get_study_user_attrs(non_existent_study_id)
        with pytest.raises(KeyError):
            storage.get_study_user_attrs(non_existent_study_id)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_set_and_get_study_system_attrs(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study()

        def check_set_and_get(key: str, value: Any) -> None:

            storage.set_study_system_attr(study_id, key, value)
            assert storage.get_study_system_attrs(study_id)[key] == value

        # Test setting value.
        for key, value in EXAMPLE_ATTRS.items():
            check_set_and_get(key, value)
        assert storage.get_study_system_attrs(study_id) == EXAMPLE_ATTRS

        # Test overwriting value.
        check_set_and_get("dataset", "ImageNet")

        # Non-existent study id.
        with pytest.raises(KeyError):
            storage.set_study_system_attr(study_id + 1, "key", "value")


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_study_user_and_system_attrs_confusion(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study()
        for key, value in EXAMPLE_ATTRS.items():
            storage.set_study_system_attr(study_id, key, value)
        assert storage.get_study_system_attrs(study_id) == EXAMPLE_ATTRS
        assert storage.get_study_user_attrs(study_id) == {}

        study_id = storage.create_new_study()
        for key, value in EXAMPLE_ATTRS.items():
            storage.set_study_user_attr(study_id, key, value)
        assert storage.get_study_user_attrs(study_id) == EXAMPLE_ATTRS
        assert storage.get_study_system_attrs(study_id) == {}


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_create_new_trial(storage_mode: str) -> None:
    def _check_trials(
        trials: List[FrozenTrial],
        idx: int,
        trial_id: int,
        time_before_creation: datetime,
        time_after_creation: datetime,
    ) -> None:
        assert len(trials) == idx + 1
        assert len({t._trial_id for t in trials}) == idx + 1
        assert trial_id in {t._trial_id for t in trials}
        assert {t.number for t in trials} == set(range(idx + 1))
        assert all(t.state == TrialState.RUNNING for t in trials)
        assert all(t.params == {} for t in trials)
        assert all(t.intermediate_values == {} for t in trials)
        assert all(t.user_attrs == {} for t in trials)
        assert all(t.system_attrs == {} for t in trials)
        assert all(
            t.datetime_start < time_before_creation
            for t in trials
            if t._trial_id != trial_id and t.datetime_start is not None
        )
        assert all(
            time_before_creation < t.datetime_start < time_after_creation
            for t in trials
            if t._trial_id == trial_id and t.datetime_start is not None
        )
        assert all(t.datetime_complete is None for t in trials)
        assert all(t.value is None for t in trials)

    with StorageSupplier(storage_mode) as storage:

        study_id = storage.create_new_study()
        n_trial_in_study = 3
        for i in range(n_trial_in_study):
            time_before_creation = datetime.now()
            trial_id = storage.create_new_trial(study_id)
            time_after_creation = datetime.now()

            trials = storage.get_all_trials(study_id)
            _check_trials(trials, i, trial_id, time_before_creation, time_after_creation)

        # Create trial in non-existent study.
        with pytest.raises(KeyError):
            storage.create_new_trial(study_id + 1)

        study_id2 = storage.create_new_study()
        for i in range(n_trial_in_study):
            storage.create_new_trial(study_id2)

            trials = storage.get_all_trials(study_id2)
            # Check that the offset of trial.number is zero.
            assert {t.number for t in trials} == set(range(i + 1))

        trials = storage.get_all_trials(study_id) + storage.get_all_trials(study_id2)
        # Check trial_ids are unique across studies.
        assert len({t._trial_id for t in trials}) == 2 * n_trial_in_study


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_create_new_trial_with_template_trial(storage_mode: str) -> None:

    start_time = datetime.now()
    complete_time = datetime.now()
    template_trial = FrozenTrial(
        state=TrialState.COMPLETE,
        value=10000,
        datetime_start=start_time,
        datetime_complete=complete_time,
        params={"x": 0.5},
        distributions={"x": UniformDistribution(0, 1)},
        user_attrs={"foo": "bar"},
        system_attrs={"baz": 123,},
        intermediate_values={1: 10, 2: 100, 3: 1000},
        number=55,  # This entry is ignored.
        trial_id=-1,  # dummy value (unused).
    )

    def _check_trials(trials: List[FrozenTrial], idx: int, trial_id: int) -> None:
        assert len(trials) == idx + 1
        assert len({t._trial_id for t in trials}) == idx + 1
        assert trial_id in {t._trial_id for t in trials}
        assert {t.number for t in trials} == set(range(idx + 1))
        assert all(t.state == template_trial.state for t in trials)
        assert all(t.params == template_trial.params for t in trials)
        assert all(t.distributions == template_trial.distributions for t in trials)
        assert all(t.intermediate_values == template_trial.intermediate_values for t in trials)
        assert all(t.user_attrs == template_trial.user_attrs for t in trials)
        assert all(t.system_attrs == template_trial.system_attrs for t in trials)
        assert all(t.datetime_start == template_trial.datetime_start for t in trials)
        assert all(t.datetime_complete == template_trial.datetime_complete for t in trials)
        assert all(t.value == template_trial.value for t in trials)

    with StorageSupplier(storage_mode) as storage:

        study_id = storage.create_new_study()

        n_trial_in_study = 3
        for i in range(n_trial_in_study):
            trial_id = storage.create_new_trial(study_id, template_trial=template_trial)
            trials = storage.get_all_trials(study_id)
            _check_trials(trials, i, trial_id)

        # Create trial in non-existent study.
        with pytest.raises(KeyError):
            storage.create_new_trial(study_id + 1)

        study_id2 = storage.create_new_study()
        for i in range(n_trial_in_study):
            storage.create_new_trial(study_id2, template_trial=template_trial)
            trials = storage.get_all_trials(study_id2)
            assert {t.number for t in trials} == set(range(i + 1))

        trials = storage.get_all_trials(study_id) + storage.get_all_trials(study_id2)
        # Check trial_ids are unique across studies.
        assert len({t._trial_id for t in trials}) == 2 * n_trial_in_study


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_trial_number_from_id(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        storage = optuna.storages.get_storage(storage)

        # Check if trial_number starts from 0.
        study_id = storage.create_new_study()

        trial_id = storage.create_new_trial(study_id)
        assert storage.get_trial_number_from_id(trial_id) == 0

        trial_id = storage.create_new_trial(study_id)
        assert storage.get_trial_number_from_id(trial_id) == 1


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_set_trial_state(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:

        study_id = storage.create_new_study()
        trial_ids = [storage.create_new_trial(study_id) for _ in ALL_STATES]

        for trial_id, state in zip(trial_ids, ALL_STATES):
            if state == TrialState.WAITING:
                continue
            assert storage.get_trial(trial_id).state == TrialState.RUNNING
            if state.is_finished():
                storage.set_trial_value(trial_id, 0.0)
            storage.set_trial_state(trial_id, state)
            assert storage.get_trial(trial_id).state == state
            if state.is_finished():
                assert storage.get_trial(trial_id).datetime_complete is not None
            else:
                assert storage.get_trial(trial_id).datetime_complete is None

        for state in ALL_STATES:
            if not state.is_finished():
                continue
            trial_id = storage.create_new_trial(study_id)
            storage.set_trial_value(trial_id, 0.0)
            storage.set_trial_state(trial_id, state)
            for state2 in ALL_STATES:
                # Cannot update states of finished trials.
                with pytest.raises(RuntimeError):
                    storage.set_trial_state(trial_id, state2)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_trial_param_and_get_trial_params(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        _, study_to_trials = _setup_studies(storage, n_study=2, n_trial=5, seed=1)

        for _, trial_id_to_trial in study_to_trials.items():
            for trial_id, expected_trial in trial_id_to_trial.items():
                assert storage.get_trial_params(trial_id) == expected_trial.params
                for key in expected_trial.params.keys():
                    assert storage.get_trial_param(trial_id, key) == expected_trial.distributions[
                        key
                    ].to_internal_repr(expected_trial.params[key])

        non_existent_trial_id = (
            max(tid for ts in study_to_trials.values() for tid in ts.keys()) + 1
        )
        with pytest.raises(KeyError):
            storage.get_trial_params(non_existent_trial_id)
        with pytest.raises(KeyError):
            storage.get_trial_param(non_existent_trial_id, "paramA")
        existent_trial_id = non_existent_trial_id - 1
        with pytest.raises(KeyError):
            storage.get_trial_param(existent_trial_id, "dummy-key")


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_set_trial_param(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:

        # Setup test across multiple studies and trials.
        study_id = storage.create_new_study()
        trial_id_1 = storage.create_new_trial(study_id)
        trial_id_2 = storage.create_new_trial(study_id)
        trial_id_3 = storage.create_new_trial(storage.create_new_study())

        # Setup distributions.
        distribution_x = UniformDistribution(low=1.0, high=2.0)
        distribution_y_1 = CategoricalDistribution(choices=("Shibuya", "Ebisu", "Meguro"))
        distribution_y_2 = CategoricalDistribution(choices=("Shibuya", "Shinsen"))
        distribution_z = LogUniformDistribution(low=1.0, high=100.0)

        # Set new params.
        assert storage.set_trial_param(trial_id_1, "x", 0.5, distribution_x)
        assert storage.set_trial_param(trial_id_1, "y", 2, distribution_y_1)
        assert storage.get_trial_param(trial_id_1, "x") == 0.5
        assert storage.get_trial_param(trial_id_1, "y") == 2
        # Check set_param breaks neither get_trial nor get_trial_params.
        assert storage.get_trial(trial_id_1).params == {"x": 0.5, "y": "Meguro"}
        assert storage.get_trial_params(trial_id_1) == {"x": 0.5, "y": "Meguro"}
        # Duplicated registration should return False.
        assert not storage.set_trial_param(trial_id_1, "x", 0.6, distribution_x)
        # Duplicated registration should not change the existing value.
        assert storage.get_trial_param(trial_id_1, "x") == 0.5

        # Set params to another trial.
        assert storage.set_trial_param(trial_id_2, "x", 0.3, distribution_x)
        assert storage.set_trial_param(trial_id_2, "z", 0.1, distribution_z)
        assert storage.get_trial_param(trial_id_2, "x") == 0.3
        assert storage.get_trial_param(trial_id_2, "z") == 0.1
        assert storage.get_trial(trial_id_2).params == {"x": 0.3, "z": 0.1}
        assert storage.get_trial_params(trial_id_2) == {"x": 0.3, "z": 0.1}

        # Set params with distributions that do not match previous ones.
        with pytest.raises(ValueError):
            storage.set_trial_param(trial_id_2, "x", 0.5, distribution_z)
        with pytest.raises(ValueError):
            storage.set_trial_param(trial_id_2, "y", 0.5, distribution_z)
        # Choices in CategoricalDistribution should match including its order.
        with pytest.raises(ValueError):
            storage.set_trial_param(
                trial_id_2, "y", 2, CategoricalDistribution(choices=("Meguro", "Shibuya", "Ebisu"))
            )

        storage.set_trial_state(trial_id_2, TrialState.COMPLETE)
        # Cannot assign params to finished trial.
        with pytest.raises(RuntimeError):
            storage.set_trial_param(trial_id_2, "y", 2, distribution_y_1)
        # Check the previous call does not change the params.
        with pytest.raises(KeyError):
            storage.get_trial_param(trial_id_2, "y")
        # State should be checked prior to distribution compatibility.
        with pytest.raises(RuntimeError):
            storage.set_trial_param(trial_id_2, "y", 0.4, distribution_z)

        # Set params of trials in a different study.
        assert storage.set_trial_param(trial_id_3, "y", 1, distribution_y_2)
        assert storage.get_trial_param(trial_id_3, "y") == 1
        assert storage.get_trial(trial_id_3).params == {"y": "Shinsen"}
        assert storage.get_trial_params(trial_id_3) == {"y": "Shinsen"}

        # Set params of non-existent trial.
        non_existent_trial_id = max([trial_id_1, trial_id_2, trial_id_3]) + 1
        with pytest.raises(KeyError):
            storage.set_trial_param(non_existent_trial_id, "x", 0.1, distribution_x)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_set_trial_value(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:

        # Setup test across multiple studies and trials.
        study_id = storage.create_new_study()
        trial_id_1 = storage.create_new_trial(study_id)
        trial_id_2 = storage.create_new_trial(study_id)
        trial_id_3 = storage.create_new_trial(storage.create_new_study())

        # Test setting new value.
        storage.set_trial_value(trial_id_1, 0.5)
        storage.set_trial_value(trial_id_3, float("inf"))

        assert storage.get_trial(trial_id_1).value == 0.5
        assert storage.get_trial(trial_id_2).value is None
        assert storage.get_trial(trial_id_3).value == float("inf")

        # Values can be overwritten.
        storage.set_trial_value(trial_id_1, 0.2)
        assert storage.get_trial(trial_id_1).value == 0.2

        non_existent_trial_id = max(trial_id_1, trial_id_2, trial_id_3) + 1
        with pytest.raises(KeyError):
            storage.set_trial_value(non_existent_trial_id, 1)

        storage.set_trial_state(trial_id_1, TrialState.COMPLETE)
        # Cannot change values of finished trials.
        with pytest.raises(RuntimeError):
            storage.set_trial_value(trial_id_1, 1)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_set_trial_intermediate_value(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:

        # Setup test across multiple studies and trials.
        study_id = storage.create_new_study()
        trial_id_1 = storage.create_new_trial(study_id)
        trial_id_2 = storage.create_new_trial(study_id)
        trial_id_3 = storage.create_new_trial(storage.create_new_study())

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

        non_existent_trial_id = max(trial_id_1, trial_id_2, trial_id_3) + 1
        with pytest.raises(KeyError):
            storage.set_trial_intermediate_value(non_existent_trial_id, 0, 0.2)

        storage.set_trial_state(trial_id_1, TrialState.COMPLETE)
        # Cannot change values of finished trials.
        with pytest.raises(RuntimeError):
            storage.set_trial_intermediate_value(trial_id_1, 0, 0.2)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_trial_user_attrs(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        _, study_to_trials = _setup_studies(storage, n_study=2, n_trial=5, seed=10)
        assert all(
            storage.get_trial_user_attrs(trial_id) == trial.user_attrs
            for trials in study_to_trials.values()
            for trial_id, trial in trials.items()
        )

        non_existent_trial = max(tid for ts in study_to_trials.values() for tid in ts.keys()) + 1
        with pytest.raises(KeyError):
            storage.get_trial_user_attrs(non_existent_trial)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_set_trial_user_attr(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        trial_id_1 = storage.create_new_trial(storage.create_new_study())

        def check_set_and_get(trial_id: int, key: str, value: Any) -> None:

            storage.set_trial_user_attr(trial_id, key, value)
            assert storage.get_trial(trial_id).user_attrs[key] == value

        # Test setting value.
        for key, value in EXAMPLE_ATTRS.items():
            check_set_and_get(trial_id_1, key, value)
        assert storage.get_trial(trial_id_1).user_attrs == EXAMPLE_ATTRS

        # Test overwriting value.
        check_set_and_get(trial_id_1, "dataset", "ImageNet")

        # Test another trial.
        trial_id_2 = storage.create_new_trial(storage.create_new_study())
        check_set_and_get(trial_id_2, "baseline_score", 0.001)
        assert len(storage.get_trial(trial_id_2).user_attrs) == 1
        assert storage.get_trial(trial_id_2).user_attrs["baseline_score"] == 0.001

        # Cannot set attributes of non-existent trials.
        non_existent_trial_id = max({trial_id_1, trial_id_2}) + 1
        with pytest.raises(KeyError):
            storage.set_trial_user_attr(non_existent_trial_id, "key", "value")

        # Cannot set attributes of finished trials.
        storage.set_trial_state(trial_id_1, TrialState.COMPLETE)
        with pytest.raises(RuntimeError):
            storage.set_trial_user_attr(trial_id_1, "key", "value")


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_trial_system_attrs(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        _, study_to_trials = _setup_studies(storage, n_study=2, n_trial=5, seed=10)
        assert all(
            storage.get_trial_system_attrs(trial_id) == trial.system_attrs
            for trials in study_to_trials.values()
            for trial_id, trial in trials.items()
        )

        non_existent_trial = max(tid for ts in study_to_trials.values() for tid in ts.keys()) + 1
        with pytest.raises(KeyError):
            storage.get_trial_system_attrs(non_existent_trial)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_set_trial_system_attr(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study()
        trial_id_1 = storage.create_new_trial(study_id)

        def check_set_and_get(trial_id: int, key: str, value: Any) -> None:

            storage.set_trial_system_attr(trial_id, key, value)
            assert storage.get_trial_system_attrs(trial_id)[key] == value

        # Test setting value.
        for key, value in EXAMPLE_ATTRS.items():
            check_set_and_get(trial_id_1, key, value)
        system_attrs = storage.get_trial(trial_id_1).system_attrs
        assert system_attrs == EXAMPLE_ATTRS

        # Test overwriting value.
        check_set_and_get(trial_id_1, "dataset", "ImageNet")

        # Test another trial.
        trial_id_2 = storage.create_new_trial(study_id)
        check_set_and_get(trial_id_2, "baseline_score", 0.001)
        system_attrs = storage.get_trial(trial_id_2).system_attrs
        assert system_attrs == {"baseline_score": 0.001}

        # Cannot set attributes of non-existent trials.
        non_existent_trial_id = max({trial_id_1, trial_id_2}) + 1
        with pytest.raises(KeyError):
            storage.set_trial_system_attr(non_existent_trial_id, "key", "value")

        # Cannot set attributes of finished trials.
        storage.set_trial_state(trial_id_1, TrialState.COMPLETE)
        with pytest.raises(RuntimeError):
            storage.set_trial_system_attr(trial_id_1, "key", "value")


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_all_study_summaries(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        expected_summaries, _ = _setup_studies(storage, n_study=10, n_trial=10, seed=46)
        summaries = storage.get_all_study_summaries()
        assert len(summaries) == len(expected_summaries)
        for _, expected_summary in expected_summaries.items():
            summary = None  # type: Optional[StudySummary]
            for s in summaries:
                if s.study_name == expected_summary.study_name:
                    summary = s
                    break
            assert summary is not None
            assert summary.direction == expected_summary.direction
            assert summary.datetime_start == expected_summary.datetime_start
            assert summary.study_name == expected_summary.study_name
            assert summary.n_trials == expected_summary.n_trials
            assert summary.user_attrs == expected_summary.user_attrs
            assert summary.system_attrs == expected_summary.system_attrs
            if expected_summary.best_trial is not None:
                assert summary.best_trial is not None
                assert summary.best_trial == expected_summary.best_trial


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_trial(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        _, study_to_trials = _setup_studies(storage, n_study=2, n_trial=20, seed=47)

        for _, expected_trials in study_to_trials.items():
            for expected_trial in expected_trials.values():
                trial = storage.get_trial(expected_trial._trial_id)
                assert trial == expected_trial

        non_existent_trial_id = (
            max(tid for ts in study_to_trials.values() for tid in ts.keys()) + 1
        )
        with pytest.raises(KeyError):
            storage.get_trial(non_existent_trial_id)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_all_trials(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        _, study_to_trials = _setup_studies(storage, n_study=2, n_trial=20, seed=48)

        for study_id, expected_trials in study_to_trials.items():
            trials = storage.get_all_trials(study_id)
            for trial in trials:
                expected_trial = expected_trials[trial._trial_id]
                assert trial == expected_trial

        non_existent_study_id = max(study_to_trials.keys()) + 1
        with pytest.raises(KeyError):
            storage.get_all_trials(non_existent_study_id)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_all_trials_deepcopy_option(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study_summaries, study_to_trials = _setup_studies(storage, n_study=2, n_trial=5, seed=49)

        for study_id in study_summaries:
            with patch("copy.deepcopy", wraps=copy.deepcopy) as mock_object:
                trials0 = storage.get_all_trials(study_id, deepcopy=True)
                assert mock_object.call_count > 0
                assert len(trials0) == len(study_to_trials[study_id])

            # Check modifying output does not break the internal state of the storage.
            trials0_original = copy.deepcopy(trials0)
            trials0[0].params["x"] = 0.1

            with patch("copy.deepcopy", wraps=copy.deepcopy) as mock_object:
                trials1 = storage.get_all_trials(study_id, deepcopy=False)
                assert mock_object.call_count == 0
                assert trials0_original == trials1


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_n_trials(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study_id_to_summaries, _ = _setup_studies(storage, n_study=2, n_trial=7, seed=50)
        for study_id in study_id_to_summaries.keys():
            assert storage.get_n_trials(study_id) == 7

        non_existent_study_id = max(study_id_to_summaries.keys()) + 1
        with pytest.raises(KeyError):
            assert storage.get_n_trials(non_existent_study_id)


def _setup_studies(
    storage: BaseStorage,
    n_study: int,
    n_trial: int,
    seed: int,
    direction: Optional[StudyDirection] = None,
) -> Tuple[Dict[int, StudySummary], Dict[int, Dict[int, FrozenTrial]]]:
    generator = random.Random(seed)
    study_id_to_summary = {}  # type: Dict[int, StudySummary]
    study_id_to_trials = {}  # type: Dict[int, Dict[int, FrozenTrial]]
    for i in range(n_study):
        study_name = "test-study-name-{}".format(i)
        study_id = storage.create_new_study(study_name=study_name)
        if direction is None:
            direction = generator.choice([StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
        storage.set_study_direction(study_id, direction)
        best_trial = None
        trials = {}
        datetime_start = None
        for j in range(n_trial):
            trial = _generate_trial(generator)
            trial.number = j
            trial._trial_id = storage.create_new_trial(study_id, trial)
            trials[trial._trial_id] = trial
            if datetime_start is None:
                datetime_start = trial.datetime_start
            else:
                datetime_start = min(datetime_start, trial.datetime_start)
            if trial.state == TrialState.COMPLETE and trial.value is not None:
                if best_trial is None:
                    best_trial = trial
                else:
                    if direction == StudyDirection.MINIMIZE and trial.value < best_trial.value:
                        best_trial = trial
                    elif direction == StudyDirection.MAXIMIZE and best_trial.value < trial.value:
                        best_trial = trial
        study_id_to_trials[study_id] = trials
        study_id_to_summary[study_id] = StudySummary(
            study_name=study_name,
            direction=direction,
            best_trial=best_trial,
            user_attrs={},
            system_attrs={},
            n_trials=len(trials),
            datetime_start=datetime_start,
            study_id=study_id,
        )
    return study_id_to_summary, study_id_to_trials


def _generate_trial(generator: random.Random) -> FrozenTrial:
    example_params = {
        "paramA": (generator.uniform(0, 1), UniformDistribution(0, 1)),
        "paramB": (generator.uniform(1, 2), LogUniformDistribution(1, 2)),
        "paramC": (
            generator.choice(["CatA", "CatB", "CatC"]),
            CategoricalDistribution(("CatA", "CatB", "CatC")),
        ),
        "paramD": (generator.uniform(-3, 0), UniformDistribution(-3, 0)),
        "paramE": (generator.choice([0.1, 0.2]), CategoricalDistribution((0.1, 0.2))),
    }
    example_attrs = {
        "attrA": "valueA",
        "attrB": 1,
        "attrC": None,
        "attrD": {"baseline_score": 0.001, "tags": ["image", "classification"]},
    }
    state = generator.choice(ALL_STATES)
    params = {}
    distributions = {}
    user_attrs = {}
    system_attrs = {}
    intermediate_values = {}
    for key, (value, dist) in example_params.items():
        if generator.choice([True, False]):
            params[key] = value
            distributions[key] = dist
    for key, value in example_attrs.items():
        if generator.choice([True, False]):
            user_attrs["usr_" + key] = value
        if generator.choice([True, False]):
            system_attrs["sys_" + key] = value
    for i in range(generator.randint(4, 10)):
        if generator.choice([True, False]):
            intermediate_values[i] = generator.uniform(-10, 10)
    return FrozenTrial(
        number=0,  # dummy
        state=state,
        value=generator.uniform(-10, 10),
        datetime_start=datetime.now(),
        datetime_complete=datetime.now() if state.is_finished() else None,
        params=params,
        distributions=distributions,
        user_attrs=user_attrs,
        system_attrs=system_attrs,
        intermediate_values=intermediate_values,
        trial_id=0,  # dummy
    )
