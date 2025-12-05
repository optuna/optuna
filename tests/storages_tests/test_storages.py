from __future__ import annotations

from collections.abc import Generator

import pytest
from pytest import FixtureRequest

import optuna
from optuna.storages import _CachedStorage
from optuna.storages import BaseStorage
from optuna.storages import InMemoryStorage
from optuna.storages import RDBStorage
from optuna.study._study_direction import StudyDirection
from optuna.testing.pytest_storages import StorageTestCase
from optuna.testing.storages import STORAGE_MODES
from optuna.testing.storages import StorageSupplier


@pytest.fixture(params=STORAGE_MODES)
def storage(request: FixtureRequest) -> Generator[BaseStorage, None, None]:
    with StorageSupplier(request.param) as storage:
        yield storage


class TestStorage(StorageTestCase): ...


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
