from __future__ import annotations

import copy
from datetime import datetime
import pickle
import random
from time import sleep
from typing import Any

import numpy as np
import pytest

import optuna
from optuna._typing import JSONSerializable
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.exceptions import UpdateFinishedTrialError
from optuna.storages import _CachedStorage
from optuna.storages import BaseStorage
from optuna.storages import GrpcStorageProxy
from optuna.storages import InMemoryStorage
from optuna.storages import RDBStorage
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.study._frozen import FrozenStudy
from optuna.study._study_direction import StudyDirection
from optuna.testing.storages import STORAGE_MODES
from optuna.testing.storages import StorageSupplier
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


ALL_STATES = list(TrialState)

EXAMPLE_ATTRS: dict[str, JSONSerializable] = {
    "dataset": "MNIST",
    "none": None,
    "json_serializable": {"baseline_score": 0.001, "tags": ["image", "classification"]},
}


def test_get_storage() -> None:
    assert isinstance(optuna.storages.get_storage(None), InMemoryStorage)
    assert isinstance(optuna.storages.get_storage("sqlite:///:memory:"), _CachedStorage)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_create_new_study(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

        frozen_studies = storage.get_all_studies()
        assert len(frozen_studies) == 1
        assert frozen_studies[0]._study_id == study_id
        assert frozen_studies[0].study_name.startswith(DEFAULT_STUDY_NAME_PREFIX)

        study_id2 = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        # Study id must be unique.
        assert study_id != study_id2
        frozen_studies = storage.get_all_studies()
        assert len(frozen_studies) == 2
        assert {s._study_id for s in frozen_studies} == {study_id, study_id2}
        assert all(s.study_name.startswith(DEFAULT_STUDY_NAME_PREFIX) for s in frozen_studies)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_create_new_study_unique_id(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        study_id2 = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        storage.delete_study(study_id2)
        study_id3 = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

        # Study id must not be reused after deletion.
        if not isinstance(storage, (RDBStorage, _CachedStorage, GrpcStorageProxy)):
            # TODO(ytsmiling) Fix RDBStorage so that it does not reuse study_id.
            assert len({study_id, study_id2, study_id3}) == 3
        frozen_studies = storage.get_all_studies()
        assert {s._study_id for s in frozen_studies} == {study_id, study_id3}


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_create_new_study_with_name(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        # Generate unique study_name from the current function name and storage_mode.
        function_name = test_create_new_study_with_name.__name__
        study_name = function_name + "/" + storage_mode
        study_id = storage.create_new_study(
            directions=[StudyDirection.MINIMIZE], study_name=study_name
        )

        assert study_name == storage.get_study_name_from_id(study_id)

        with pytest.raises(optuna.exceptions.DuplicatedStudyError):
            storage.create_new_study(directions=[StudyDirection.MINIMIZE], study_name=study_name)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_delete_study(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        storage.create_new_trial(study_id)
        trials = storage.get_all_trials(study_id)
        assert len(trials) == 1

        with pytest.raises(KeyError):
            # Deletion of non-existent study.
            storage.delete_study(study_id + 1)

        storage.delete_study(study_id)
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trials = storage.get_all_trials(study_id)
        assert len(trials) == 0

        storage.delete_study(study_id)
        with pytest.raises(KeyError):
            # Double free.
            storage.delete_study(study_id)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_delete_study_after_create_multiple_studies(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study_id1 = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        study_id2 = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        study_id3 = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

        storage.delete_study(study_id2)

        studies = {s._study_id: s for s in storage.get_all_studies()}
        assert study_id1 in studies
        assert study_id2 not in studies
        assert study_id3 in studies


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_study_id_from_name_and_get_study_name_from_id(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        # Generate unique study_name from the current function name and storage_mode.
        function_name = test_get_study_id_from_name_and_get_study_name_from_id.__name__
        study_name = function_name + "/" + storage_mode
        study_id = storage.create_new_study(
            directions=[StudyDirection.MINIMIZE], study_name=study_name
        )

        # Test existing study.
        assert storage.get_study_name_from_id(study_id) == study_name
        assert storage.get_study_id_from_name(study_name) == study_id

        # Test not existing study.
        with pytest.raises(KeyError):
            storage.get_study_id_from_name("dummy-name")

        with pytest.raises(KeyError):
            storage.get_study_name_from_id(study_id + 1)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_set_and_get_study_directions(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        for target in [
            (StudyDirection.MINIMIZE,),
            (StudyDirection.MAXIMIZE,),
            (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE),
            (StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE),
            [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE],
            [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE],
        ]:
            study_id = storage.create_new_study(directions=target)

            def check_get() -> None:
                got_directions = storage.get_study_directions(study_id)

                assert got_directions == list(
                    target
                ), "Direction of a study should be a tuple of `StudyDirection` objects."

            # Test setting value.
            check_get()

            # Test non-existent study.
            non_existent_study_id = study_id + 1

            with pytest.raises(KeyError):
                storage.get_study_directions(non_existent_study_id)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_set_and_get_study_user_attrs(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

        def check_set_and_get(key: str, value: Any) -> None:
            storage.set_study_user_attr(study_id, key, value)
            assert storage.get_study_user_attrs(study_id)[key] == value

        # Test setting value.
        for key, value in EXAMPLE_ATTRS.items():
            check_set_and_get(key, value)
        assert storage.get_study_user_attrs(study_id) == EXAMPLE_ATTRS

        # Test overwriting value.
        check_set_and_get("dataset", "ImageNet")

        # Non-existent study id.
        non_existent_study_id = study_id + 1
        with pytest.raises(KeyError):
            storage.get_study_user_attrs(non_existent_study_id)

        # Non-existent study id.
        with pytest.raises(KeyError):
            storage.set_study_user_attr(non_existent_study_id, "key", "value")


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_set_and_get_study_system_attrs(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

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
        non_existent_study_id = study_id + 1
        with pytest.raises(KeyError):
            storage.get_study_system_attrs(non_existent_study_id)

        # Non-existent study id.
        with pytest.raises(KeyError):
            storage.set_study_system_attr(non_existent_study_id, "key", "value")


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_study_user_and_system_attrs_confusion(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        for key, value in EXAMPLE_ATTRS.items():
            storage.set_study_system_attr(study_id, key, value)
        assert storage.get_study_system_attrs(study_id) == EXAMPLE_ATTRS
        assert storage.get_study_user_attrs(study_id) == {}

        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        for key, value in EXAMPLE_ATTRS.items():
            storage.set_study_user_attr(study_id, key, value)
        assert storage.get_study_user_attrs(study_id) == EXAMPLE_ATTRS
        assert storage.get_study_system_attrs(study_id) == {}


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_create_new_trial(storage_mode: str) -> None:
    def _check_trials(
        trials: list[FrozenTrial],
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
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        n_trial_in_study = 3
        for i in range(n_trial_in_study):
            time_before_creation = datetime.now()
            sleep(0.001)  # Sleep 1ms to avoid faulty assertion on Windows OS.
            trial_id = storage.create_new_trial(study_id)
            sleep(0.001)
            time_after_creation = datetime.now()

            trials = storage.get_all_trials(study_id)
            _check_trials(trials, i, trial_id, time_before_creation, time_after_creation)

        # Create trial in non-existent study.
        with pytest.raises(KeyError):
            storage.create_new_trial(study_id + 1)

        study_id2 = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        for i in range(n_trial_in_study):
            storage.create_new_trial(study_id2)

            trials = storage.get_all_trials(study_id2)
            # Check that the offset of trial.number is zero.
            assert {t.number for t in trials} == set(range(i + 1))

        trials = storage.get_all_trials(study_id) + storage.get_all_trials(study_id2)
        # Check trial_ids are unique across studies.
        assert len({t._trial_id for t in trials}) == 2 * n_trial_in_study


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize(
    "start_time,complete_time",
    [(datetime.now(), datetime.now()), (datetime(2022, 9, 1), datetime(2022, 9, 2))],
)
def test_create_new_trial_with_template_trial(
    storage_mode: str, start_time: datetime, complete_time: datetime
) -> None:
    template_trial = FrozenTrial(
        state=TrialState.COMPLETE,
        value=10000,
        datetime_start=start_time,
        datetime_complete=complete_time,
        params={"x": 0.5},
        distributions={"x": FloatDistribution(0, 1)},
        user_attrs={"foo": "bar"},
        system_attrs={"baz": 123},
        intermediate_values={1: 10, 2: 100, 3: 1000},
        number=55,  # This entry is ignored.
        trial_id=-1,  # dummy value (unused).
    )

    def _check_trials(trials: list[FrozenTrial], idx: int, trial_id: int) -> None:
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
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

        n_trial_in_study = 3
        for i in range(n_trial_in_study):
            trial_id = storage.create_new_trial(study_id, template_trial=template_trial)
            trials = storage.get_all_trials(study_id)
            _check_trials(trials, i, trial_id)

        # Create trial in non-existent study.
        with pytest.raises(KeyError):
            storage.create_new_trial(study_id + 1)

        study_id2 = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
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
        # Check if trial_number starts from 0.
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

        trial_id = storage.create_new_trial(study_id)
        assert storage.get_trial_number_from_id(trial_id) == 0

        trial_id = storage.create_new_trial(study_id)
        assert storage.get_trial_number_from_id(trial_id) == 1

        with pytest.raises(KeyError):
            storage.get_trial_number_from_id(trial_id + 1)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_set_trial_state_values_for_state(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trial_ids = [storage.create_new_trial(study_id) for _ in ALL_STATES]

        for trial_id, state in zip(trial_ids, ALL_STATES):
            if state == TrialState.WAITING:
                continue
            assert storage.get_trial(trial_id).state == TrialState.RUNNING
            datetime_start_prev = storage.get_trial(trial_id).datetime_start
            storage.set_trial_state_values(
                trial_id, state=state, values=(0.0,) if state.is_finished() else None
            )
            assert storage.get_trial(trial_id).state == state
            # Repeated state changes to RUNNING should not trigger further datetime_start changes.
            if state == TrialState.RUNNING:
                assert storage.get_trial(trial_id).datetime_start == datetime_start_prev
            if state.is_finished():
                assert storage.get_trial(trial_id).datetime_complete is not None
            else:
                assert storage.get_trial(trial_id).datetime_complete is None

        # Non-existent study.
        with pytest.raises(KeyError):
            non_existent_trial_id = max(trial_ids) + 1
            storage.set_trial_state_values(
                non_existent_trial_id,
                state=TrialState.COMPLETE,
            )

        for state in ALL_STATES:
            if not state.is_finished():
                continue
            trial_id = storage.create_new_trial(study_id)
            storage.set_trial_state_values(trial_id, state=state, values=(0.0,))
            for state2 in ALL_STATES:
                # Cannot update states of finished trials.
                with pytest.raises(UpdateFinishedTrialError):
                    storage.set_trial_state_values(trial_id, state=state2)


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
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trial_id_1 = storage.create_new_trial(study_id)
        trial_id_2 = storage.create_new_trial(study_id)
        trial_id_3 = storage.create_new_trial(
            storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        )

        # Setup distributions.
        distribution_x = FloatDistribution(low=1.0, high=2.0)
        distribution_y_1 = CategoricalDistribution(choices=("Shibuya", "Ebisu", "Meguro"))
        distribution_y_2 = CategoricalDistribution(choices=("Shibuya", "Shinsen"))
        distribution_z = FloatDistribution(low=1.0, high=100.0, log=True)

        # Set new params.
        storage.set_trial_param(trial_id_1, "x", 0.5, distribution_x)
        storage.set_trial_param(trial_id_1, "y", 2, distribution_y_1)
        assert storage.get_trial_param(trial_id_1, "x") == 0.5
        assert storage.get_trial_param(trial_id_1, "y") == 2
        # Check set_param breaks neither get_trial nor get_trial_params.
        assert storage.get_trial(trial_id_1).params == {"x": 0.5, "y": "Meguro"}
        assert storage.get_trial_params(trial_id_1) == {"x": 0.5, "y": "Meguro"}

        # Set params to another trial.
        storage.set_trial_param(trial_id_2, "x", 0.3, distribution_x)
        storage.set_trial_param(trial_id_2, "z", 0.1, distribution_z)
        assert storage.get_trial_param(trial_id_2, "x") == 0.3
        assert storage.get_trial_param(trial_id_2, "z") == 0.1
        assert storage.get_trial(trial_id_2).params == {"x": 0.3, "z": 0.1}
        assert storage.get_trial_params(trial_id_2) == {"x": 0.3, "z": 0.1}

        # Set params with distributions that do not match previous ones.
        with pytest.raises(ValueError):
            storage.set_trial_param(trial_id_2, "y", 0.5, distribution_z)
        # Choices in CategoricalDistribution should match including its order.
        with pytest.raises(ValueError):
            storage.set_trial_param(
                trial_id_2, "y", 2, CategoricalDistribution(choices=("Meguro", "Shibuya", "Ebisu"))
            )

        storage.set_trial_state_values(trial_id_2, state=TrialState.COMPLETE)
        # Cannot assign params to finished trial.
        with pytest.raises(UpdateFinishedTrialError):
            storage.set_trial_param(trial_id_2, "y", 2, distribution_y_1)
        # Check the previous call does not change the params.
        with pytest.raises(KeyError):
            storage.get_trial_param(trial_id_2, "y")
        # State should be checked prior to distribution compatibility.
        with pytest.raises(UpdateFinishedTrialError):
            storage.set_trial_param(trial_id_2, "y", 0.4, distribution_z)

        # Set params of trials in a different study.
        storage.set_trial_param(trial_id_3, "y", 1, distribution_y_2)
        assert storage.get_trial_param(trial_id_3, "y") == 1
        assert storage.get_trial(trial_id_3).params == {"y": "Shinsen"}
        assert storage.get_trial_params(trial_id_3) == {"y": "Shinsen"}

        # Set params of non-existent trial.
        non_existent_trial_id = max([trial_id_1, trial_id_2, trial_id_3]) + 1
        with pytest.raises(KeyError):
            storage.set_trial_param(non_existent_trial_id, "x", 0.1, distribution_x)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_set_trial_state_values_for_values(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        # Setup test across multiple studies and trials.
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trial_id_1 = storage.create_new_trial(study_id)
        trial_id_2 = storage.create_new_trial(study_id)
        trial_id_3 = storage.create_new_trial(
            storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        )
        trial_id_4 = storage.create_new_trial(study_id)
        trial_id_5 = storage.create_new_trial(study_id)

        # Test setting new value.
        storage.set_trial_state_values(trial_id_1, state=TrialState.COMPLETE, values=(0.5,))
        storage.set_trial_state_values(
            trial_id_3, state=TrialState.COMPLETE, values=(float("inf"),)
        )
        storage.set_trial_state_values(
            trial_id_4, state=TrialState.WAITING, values=(0.1, 0.2, 0.3)
        )
        storage.set_trial_state_values(
            trial_id_5, state=TrialState.WAITING, values=[0.1, 0.2, 0.3]
        )

        assert storage.get_trial(trial_id_1).value == 0.5
        assert storage.get_trial(trial_id_2).value is None
        assert storage.get_trial(trial_id_3).value == float("inf")
        assert storage.get_trial(trial_id_4).values == [0.1, 0.2, 0.3]
        assert storage.get_trial(trial_id_5).values == [0.1, 0.2, 0.3]

        non_existent_trial_id = max(trial_id_1, trial_id_2, trial_id_3, trial_id_4, trial_id_5) + 1
        with pytest.raises(KeyError):
            storage.set_trial_state_values(
                non_existent_trial_id, state=TrialState.COMPLETE, values=(1,)
            )

        # Cannot change values of finished trials.
        with pytest.raises(UpdateFinishedTrialError):
            storage.set_trial_state_values(trial_id_1, state=TrialState.COMPLETE, values=(1,))


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_set_trial_intermediate_value(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        # Setup test across multiple studies and trials.
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trial_id_1 = storage.create_new_trial(study_id)
        trial_id_2 = storage.create_new_trial(study_id)
        trial_id_3 = storage.create_new_trial(
            storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        )
        trial_id_4 = storage.create_new_trial(study_id)

        # Test setting new values.
        storage.set_trial_intermediate_value(trial_id_1, 0, 0.3)
        storage.set_trial_intermediate_value(trial_id_1, 2, 0.4)
        storage.set_trial_intermediate_value(trial_id_3, 0, 0.1)
        storage.set_trial_intermediate_value(trial_id_3, 1, 0.4)
        storage.set_trial_intermediate_value(trial_id_3, 2, 0.5)
        storage.set_trial_intermediate_value(trial_id_3, 3, float("inf"))
        storage.set_trial_intermediate_value(trial_id_4, 0, float("nan"))

        assert storage.get_trial(trial_id_1).intermediate_values == {0: 0.3, 2: 0.4}
        assert storage.get_trial(trial_id_2).intermediate_values == {}
        assert storage.get_trial(trial_id_3).intermediate_values == {
            0: 0.1,
            1: 0.4,
            2: 0.5,
            3: float("inf"),
        }
        assert np.isnan(storage.get_trial(trial_id_4).intermediate_values[0])

        # Test setting existing step.
        storage.set_trial_intermediate_value(trial_id_1, 0, 0.2)
        assert storage.get_trial(trial_id_1).intermediate_values == {0: 0.2, 2: 0.4}

        non_existent_trial_id = max(trial_id_1, trial_id_2, trial_id_3, trial_id_4) + 1
        with pytest.raises(KeyError):
            storage.set_trial_intermediate_value(non_existent_trial_id, 0, 0.2)

        storage.set_trial_state_values(trial_id_1, state=TrialState.COMPLETE)
        # Cannot change values of finished trials.
        with pytest.raises(UpdateFinishedTrialError):
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
        trial_id_1 = storage.create_new_trial(
            storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        )

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
        trial_id_2 = storage.create_new_trial(
            storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        )
        check_set_and_get(trial_id_2, "baseline_score", 0.001)
        assert len(storage.get_trial(trial_id_2).user_attrs) == 1
        assert storage.get_trial(trial_id_2).user_attrs["baseline_score"] == 0.001

        # Cannot set attributes of non-existent trials.
        non_existent_trial_id = max({trial_id_1, trial_id_2}) + 1
        with pytest.raises(KeyError):
            storage.set_trial_user_attr(non_existent_trial_id, "key", "value")

        # Cannot set attributes of finished trials.
        storage.set_trial_state_values(trial_id_1, state=TrialState.COMPLETE)
        with pytest.raises(UpdateFinishedTrialError):
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
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
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
        storage.set_trial_state_values(trial_id_1, state=TrialState.COMPLETE)
        with pytest.raises(UpdateFinishedTrialError):
            storage.set_trial_system_attr(trial_id_1, "key", "value")


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_all_studies(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        expected_frozen_studies, _ = _setup_studies(storage, n_study=10, n_trial=10, seed=46)
        frozen_studies = storage.get_all_studies()
        assert len(frozen_studies) == len(expected_frozen_studies)
        for _, expected_frozen_study in expected_frozen_studies.items():
            frozen_study: FrozenStudy | None = None
            for s in frozen_studies:
                if s.study_name == expected_frozen_study.study_name:
                    frozen_study = s
                    break
            assert frozen_study is not None
            assert frozen_study.direction == expected_frozen_study.direction
            assert frozen_study.study_name == expected_frozen_study.study_name
            assert frozen_study.user_attrs == expected_frozen_study.user_attrs
            assert frozen_study.system_attrs == expected_frozen_study.system_attrs


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
@pytest.mark.parametrize("param_names", [["a", "b"], ["b", "a"]])
def test_get_all_trials_params_order(storage_mode: str, param_names: list[str]) -> None:
    # We don't actually require that all storages to preserve the order of parameters,
    # but all current implementations except for GrpcStorageProxy do, so we test this property.
    if storage_mode in ("grpc_rdb", "grpc_journal_file"):
        pytest.skip("GrpcStorageProxy does not preserve the order of parameters.")

    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trial_id = storage.create_new_trial(
            study_id, optuna.trial.create_trial(state=TrialState.RUNNING)
        )
        for param_name in param_names:
            storage.set_trial_param(
                trial_id, param_name, 1.0, distribution=FloatDistribution(0.0, 2.0)
            )

        trials = storage.get_all_trials(study_id)
        assert list(trials[0].params.keys()) == param_names


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_all_trials_deepcopy_option(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        frozen_studies, study_to_trials = _setup_studies(storage, n_study=2, n_trial=5, seed=49)

        for study_id in frozen_studies:
            trials0 = storage.get_all_trials(study_id, deepcopy=True)
            assert len(trials0) == len(study_to_trials[study_id])

            # Check modifying output does not break the internal state of the storage.
            trials0_original = copy.deepcopy(trials0)
            trials0[0].params["x"] = 0.1

            trials1 = storage.get_all_trials(study_id, deepcopy=False)
            assert trials0_original == trials1


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_all_trials_state_option(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study(directions=[StudyDirection.MAXIMIZE])
        generator = random.Random(51)

        states = (
            TrialState.COMPLETE,
            TrialState.COMPLETE,
            TrialState.PRUNED,
        )

        for state in states:
            t = _generate_trial(generator)
            t.state = state
            storage.create_new_trial(study_id, template_trial=t)

        trials = storage.get_all_trials(study_id, states=None)
        assert len(trials) == 3

        trials = storage.get_all_trials(study_id, states=(TrialState.COMPLETE,))
        assert len(trials) == 2
        assert all(t.state == TrialState.COMPLETE for t in trials)

        trials = storage.get_all_trials(study_id, states=(TrialState.COMPLETE, TrialState.PRUNED))
        assert len(trials) == 3
        assert all(t.state in (TrialState.COMPLETE, TrialState.PRUNED) for t in trials)

        trials = storage.get_all_trials(study_id, states=())
        assert len(trials) == 0

        other_states = [
            s for s in ALL_STATES if s != TrialState.COMPLETE and s != TrialState.PRUNED
        ]
        for state in other_states:
            trials = storage.get_all_trials(study_id, states=(state,))
            assert len(trials) == 0


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_all_trials_not_modified(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        _, study_to_trials = _setup_studies(storage, n_study=2, n_trial=20, seed=48)

        for study_id in study_to_trials.keys():
            trials = storage.get_all_trials(study_id, deepcopy=False)
            deepcopied_trials = copy.deepcopy(trials)

            for trial in trials:
                if not trial.state.is_finished():
                    storage.set_trial_param(trial._trial_id, "paramX", 0, FloatDistribution(0, 1))
                    storage.set_trial_user_attr(trial._trial_id, "usr_attrX", 0)
                    storage.set_trial_system_attr(trial._trial_id, "sys_attrX", 0)

                if trial.state == TrialState.RUNNING:
                    if trial.number % 3 == 0:
                        storage.set_trial_state_values(trial._trial_id, TrialState.COMPLETE, [0])
                    elif trial.number % 3 == 1:
                        storage.set_trial_intermediate_value(trial._trial_id, 0, 0)
                        storage.set_trial_state_values(trial._trial_id, TrialState.PRUNED, [0])
                    else:
                        storage.set_trial_state_values(trial._trial_id, TrialState.FAIL)
                elif trial.state == TrialState.WAITING:
                    storage.set_trial_state_values(trial._trial_id, TrialState.RUNNING)

            assert trials == deepcopied_trials


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_n_trials(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study_id_to_frozen_studies, _ = _setup_studies(storage, n_study=2, n_trial=7, seed=50)
        for study_id in study_id_to_frozen_studies:
            assert storage.get_n_trials(study_id) == 7

        non_existent_study_id = max(study_id_to_frozen_studies.keys()) + 1
        with pytest.raises(KeyError):
            assert storage.get_n_trials(non_existent_study_id)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_n_trials_state_option(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study(directions=(StudyDirection.MAXIMIZE,))
        generator = random.Random(51)

        states = [
            TrialState.COMPLETE,
            TrialState.COMPLETE,
            TrialState.PRUNED,
        ]

        for s in states:
            t = _generate_trial(generator)
            t.state = s
            storage.create_new_trial(study_id, template_trial=t)

        assert storage.get_n_trials(study_id, TrialState.COMPLETE) == 2
        assert storage.get_n_trials(study_id, TrialState.PRUNED) == 1

        other_states = [
            s for s in ALL_STATES if s != TrialState.COMPLETE and s != TrialState.PRUNED
        ]
        for s in other_states:
            assert storage.get_n_trials(study_id, s) == 0


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize("direction", [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE])
@pytest.mark.parametrize(
    "values",
    [
        [0.0, 1.0, 2.0],
        [0.0, float("inf"), 1.0],
        [0.0, float("-inf"), 1.0],
        [float("inf"), 0.0, 1.0, float("-inf")],
        [float("inf")],
        [float("-inf")],
    ],
)
def test_get_best_trial(storage_mode: str, direction: StudyDirection, values: list[float]) -> None:
    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study(directions=[direction])
        with pytest.raises(ValueError):
            storage.get_best_trial(study_id)

        with pytest.raises(KeyError):
            storage.get_best_trial(study_id + 1)

        generator = random.Random(51)

        for v in values:
            template_trial = _generate_trial(generator)
            template_trial.state = TrialState.COMPLETE
            template_trial.value = v
            storage.create_new_trial(study_id, template_trial=template_trial)
        expected_value = max(values) if direction == StudyDirection.MAXIMIZE else min(values)
        assert storage.get_best_trial(study_id).value == expected_value


def test_get_trials_included_trial_ids() -> None:
    storage_mode = "sqlite"

    with StorageSupplier(storage_mode) as storage:
        assert isinstance(storage, RDBStorage)
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

        trial_id = storage.create_new_trial(study_id)
        trial_id_greater_than = trial_id + 500000

        trials = storage._get_trials(
            study_id,
            states=None,
            included_trial_ids=set(),
            trial_id_greater_than=trial_id_greater_than,
        )
        assert len(trials) == 0

        # A large exclusion list used to raise errors. Check that it is not an issue.
        # See https://github.com/optuna/optuna/issues/1457.
        trials = storage._get_trials(
            study_id,
            states=None,
            included_trial_ids=set(range(500000)),
            trial_id_greater_than=trial_id_greater_than,
        )
        assert len(trials) == 1


def test_get_trials_trial_id_greater_than() -> None:
    storage_mode = "sqlite"

    with StorageSupplier(storage_mode) as storage:
        assert isinstance(storage, RDBStorage)
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

        storage.create_new_trial(study_id)

        trials = storage._get_trials(
            study_id, states=None, included_trial_ids=set(), trial_id_greater_than=-1
        )
        assert len(trials) == 1

        trials = storage._get_trials(
            study_id, states=None, included_trial_ids=set(), trial_id_greater_than=500001
        )
        assert len(trials) == 0


def _setup_studies(
    storage: BaseStorage,
    n_study: int,
    n_trial: int,
    seed: int,
    direction: StudyDirection | None = None,
) -> tuple[dict[int, FrozenStudy], dict[int, dict[int, FrozenTrial]]]:
    generator = random.Random(seed)
    study_id_to_frozen_study: dict[int, FrozenStudy] = {}
    study_id_to_trials: dict[int, dict[int, FrozenTrial]] = {}
    for i in range(n_study):
        study_name = "test-study-name-{}".format(i)
        if direction is None:
            direction = generator.choice([StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
        study_id = storage.create_new_study(directions=(direction,), study_name=study_name)
        storage.set_study_user_attr(study_id, "u", i)
        storage.set_study_system_attr(study_id, "s", i)
        trials = {}
        for j in range(n_trial):
            trial = _generate_trial(generator)
            trial.number = j
            trial._trial_id = storage.create_new_trial(study_id, trial)
            trials[trial._trial_id] = trial
        study_id_to_trials[study_id] = trials
        study_id_to_frozen_study[study_id] = FrozenStudy(
            study_name=study_name,
            direction=direction,
            user_attrs={"u": i},
            system_attrs={"s": i},
            study_id=study_id,
        )
    return study_id_to_frozen_study, study_id_to_trials


def _generate_trial(generator: random.Random) -> FrozenTrial:
    example_params = {
        "paramA": (generator.uniform(0, 1), FloatDistribution(0, 1)),
        "paramB": (generator.uniform(1, 2), FloatDistribution(1, 2, log=True)),
        "paramC": (
            generator.choice(["CatA", "CatB", "CatC"]),
            CategoricalDistribution(("CatA", "CatB", "CatC")),
        ),
        "paramD": (generator.uniform(-3, 0), FloatDistribution(-3, 0)),
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
    system_attrs: dict[str, Any] = {}
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


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_best_trial_for_multi_objective_optimization(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study(
            directions=(StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE)
        )

        generator = random.Random(51)
        for i in range(3):
            template_trial = _generate_trial(generator)
            template_trial.state = TrialState.COMPLETE
            template_trial.values = [i, i + 1]
            storage.create_new_trial(study_id, template_trial=template_trial)

        with pytest.raises(RuntimeError):
            storage.get_best_trial(study_id)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_trial_id_from_study_id_trial_number(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        with pytest.raises(KeyError):  # Matching study does not exist.
            storage.get_trial_id_from_study_id_trial_number(study_id=0, trial_number=0)

        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

        with pytest.raises(KeyError):  # Matching trial does not exist.
            storage.get_trial_id_from_study_id_trial_number(study_id, trial_number=0)

        trial_id = storage.create_new_trial(study_id)

        assert trial_id == storage.get_trial_id_from_study_id_trial_number(
            study_id, trial_number=0
        )

        # Trial IDs are globally unique within a storage but numbers are only unique within a
        # study. Create a second study within the same storage.
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

        trial_id = storage.create_new_trial(study_id)

        assert trial_id == storage.get_trial_id_from_study_id_trial_number(
            study_id, trial_number=0
        )


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_pickle_storage(storage_mode: str) -> None:
    if "redis" in storage_mode:
        pytest.skip("The `fakeredis` does not support multi instances.")

    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        storage.set_study_system_attr(study_id, "key", "pickle")

        restored_storage = pickle.loads(pickle.dumps(storage))

        storage_system_attrs = storage.get_study_system_attrs(study_id)
        restored_storage_system_attrs = restored_storage.get_study_system_attrs(study_id)
        assert storage_system_attrs == restored_storage_system_attrs == {"key": "pickle"}

        if isinstance(storage, RDBStorage):
            assert storage.url == restored_storage.url
            assert storage.engine_kwargs == restored_storage.engine_kwargs
            assert storage.skip_compatibility_check == restored_storage.skip_compatibility_check
            assert storage.engine != restored_storage.engine
            assert storage.scoped_session != restored_storage.scoped_session
            assert storage._version_manager != restored_storage._version_manager


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_check_trial_is_updatable(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
        trial_id = storage.create_new_trial(study_id)

        storage.check_trial_is_updatable(trial_id, TrialState.RUNNING)
        storage.check_trial_is_updatable(trial_id, TrialState.WAITING)

        with pytest.raises(UpdateFinishedTrialError):
            storage.check_trial_is_updatable(trial_id, TrialState.FAIL)

        with pytest.raises(UpdateFinishedTrialError):
            storage.check_trial_is_updatable(trial_id, TrialState.PRUNED)

        with pytest.raises(UpdateFinishedTrialError):
            storage.check_trial_is_updatable(trial_id, TrialState.COMPLETE)
