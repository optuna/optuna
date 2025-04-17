from __future__ import annotations

from datetime import timedelta
import random

from freezegun import freeze_time
import pytest

from optuna.study._study_direction import StudyDirection
from optuna.testing.storages import STORAGE_MODES_GRPC
from optuna.testing.storages import StorageSupplier

from .test_storages import generate_trial


@pytest.mark.parametrize("storage_mode", STORAGE_MODES_GRPC)
def test_get_all_trials_ttl_cache_is_none(storage_mode: str) -> None:
    n_trials_init = 1
    n_trials_add = 2
    with StorageSupplier(storage_mode) as storage:
        study_id = storage.create_new_study(directions=[StudyDirection.MAXIMIZE])
        generator = random.Random(51)

        for _ in range(n_trials_init):
            t = generate_trial(generator)
            storage.create_new_trial(study_id, template_trial=t)

        # Synchronize the storage cache.
        trials = storage.get_all_trials(study_id)
        assert len(trials) == n_trials_init

        for _ in range(n_trials_add):
            t = generate_trial(generator)
            storage.create_new_trial(study_id, template_trial=t)

        # Synchronize the storage cache always if ttl_cache_seconds is None.
        trials = storage.get_all_trials(study_id)
        assert len(trials) == n_trials_init + n_trials_add


@pytest.mark.parametrize("storage_mode", STORAGE_MODES_GRPC)
def test_get_all_trials_ttl_cache(storage_mode: str) -> None:
    n_trials_init = 1
    n_trials_add = 2
    ttl_cache_seconds = 10
    with StorageSupplier(storage_mode, ttl_cache_seconds=ttl_cache_seconds) as storage:
        with freeze_time() as frozen_datetime:
            study_id = storage.create_new_study(directions=[StudyDirection.MAXIMIZE])
            generator = random.Random(51)

            for _ in range(n_trials_init):
                t = generate_trial(generator)
                storage.create_new_trial(study_id, template_trial=t)

            # Synchronize the storage cache.
            trials = storage.get_all_trials(study_id)
            assert len(trials) == n_trials_init

            for _ in range(n_trials_add):
                t = generate_trial(generator)
                storage.create_new_trial(study_id, template_trial=t)

            # not synchronized yet.
            trials = storage.get_all_trials(study_id)
            assert len(trials) == n_trials_init

            frozen_datetime.tick(delta=timedelta(seconds=ttl_cache_seconds))
            # Synchronize the storage cache after the ttl_cache_seconds.
            trials = storage.get_all_trials(study_id)
            assert len(trials) == n_trials_init + n_trials_add
