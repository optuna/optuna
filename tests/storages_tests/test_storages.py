from __future__ import annotations

from collections.abc import Generator
import pickle

import pytest
from pytest import FixtureRequest

import optuna
from optuna.storages import _CachedStorage
from optuna.storages import BaseStorage
from optuna.storages import GrpcStorageProxy
from optuna.storages import InMemoryStorage
from optuna.storages import JournalStorage
from optuna.storages import RDBStorage
from optuna.storages.journal import JournalRedisBackend
from optuna.study._study_direction import StudyDirection
from optuna.testing.pytest_storages import StorageTestCase
from optuna.testing.storages import STORAGE_MODES
from optuna.testing.storages import StorageSupplier


@pytest.fixture(params=STORAGE_MODES)
def storage(request: FixtureRequest) -> Generator[BaseStorage, None, None]:
    with StorageSupplier(request.param) as storage:
        yield storage


class TestStorage(StorageTestCase):
    def test_create_new_study_unique_id(self, storage: BaseStorage) -> None:
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

    @pytest.mark.parametrize("param_names", [["a", "b"], ["b", "a"]])
    def test_get_all_trials_params_order(
        self, storage: BaseStorage, param_names: list[str]
    ) -> None:
        # We don't actually require that all storages to preserve the order of parameters,
        # but all current implementations except for GrpcStorageProxy do, so we test this property.
        if isinstance(storage, GrpcStorageProxy):
            pytest.skip("GrpcStorageProxy does not preserve the order of parameters.")

        super().test_get_all_trials_params_order(storage, param_names)

    def test_pickle_storage(self, storage: BaseStorage) -> None:
        if isinstance(storage, JournalStorage) and isinstance(
            storage._backend, JournalRedisBackend
        ):
            pytest.skip("The `fakeredis` does not support multi instances.")

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


def test_get_storage() -> None:
    assert isinstance(optuna.storages.get_storage(None), InMemoryStorage)
    assert isinstance(optuna.storages.get_storage("sqlite:///:memory:"), _CachedStorage)


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
