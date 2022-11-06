from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import pickle
import tempfile
from types import TracebackType
from typing import Any
from typing import IO
from typing import Optional
from typing import Type

import fakeredis
import pytest

import optuna
from optuna.storages._journal.file import JournalFileBaseLock


LOG_STORAGE = {
    "file_with_open_lock",
    "file_with_link_lock",
    "redis_default",
    "redis_with_use_cluster",
}


class JournalLogStorageSupplier:
    def __init__(self, storage_type: str) -> None:
        self.storage_type = storage_type
        self.tempfile: Optional[IO[Any]] = None

    def __enter__(self) -> optuna.storages.BaseJournalLogStorage:
        if self.storage_type.startswith("file"):
            self.tempfile = tempfile.NamedTemporaryFile()
            lock: JournalFileBaseLock
            if self.storage_type == "file_with_open_lock":
                lock = optuna.storages.JournalFileOpenLock(self.tempfile.name)
            elif self.storage_type == "file_with_link_lock":
                lock = optuna.storages.JournalFileSymlinkLock(self.tempfile.name)
            else:
                raise Exception("Must not reach here")
            return optuna.storages.JournalFileStorage(self.tempfile.name, lock)
        elif self.storage_type.startswith("redis"):
            use_cluster = self.storage_type == "redis_with_use_cluster"
            journal_redis_storage = optuna.storages.JournalRedisStorage(
                "redis://localhost", use_cluster
            )
            journal_redis_storage._redis = fakeredis.FakeStrictRedis()
            return journal_redis_storage
        else:
            raise RuntimeError("Unknown log storage type: {}".format(self.storage_type))

    def __exit__(
        self, exc_type: Type[BaseException], exc_val: BaseException, exc_tb: TracebackType
    ) -> None:

        if self.tempfile:
            self.tempfile.close()


@pytest.mark.parametrize("log_storage_type", LOG_STORAGE)
def test_concurrent_append_logs_for_multi_processes(log_storage_type: str) -> None:
    if log_storage_type.startswith("redis"):
        pytest.skip("The `fakeredis` does not support multi process environments.")

    num_executors = 10
    num_records = 200
    record = {"key": "value"}

    with JournalLogStorageSupplier(log_storage_type) as storage:
        with ProcessPoolExecutor(num_executors) as pool:
            pool.map(storage.append_logs, [[record] for _ in range(num_records)], timeout=20)

        assert len(storage.read_logs(0)) == num_records
        assert all(record == r for r in storage.read_logs(0))


@pytest.mark.parametrize("log_storage_type", LOG_STORAGE)
def test_concurrent_append_logs_for_multi_threads(log_storage_type: str) -> None:
    num_executors = 10
    num_records = 200
    record = {"key": "value"}

    with JournalLogStorageSupplier(log_storage_type) as storage:
        with ThreadPoolExecutor(num_executors) as pool:
            pool.map(storage.append_logs, [[record] for _ in range(num_records)], timeout=20)

        assert len(storage.read_logs(0)) == num_records
        assert all(record == r for r in storage.read_logs(0))


def pop_waiting_trial(file_path: str, study_name: str) -> Optional[int]:
    file_storage = optuna.storages.JournalFileStorage(file_path)
    storage = optuna.storages.JournalStorage(file_storage)
    study = optuna.load_study(storage=storage, study_name=study_name)
    return study._pop_waiting_trial_id()


def test_pop_waiting_trial_multiprocess_safe() -> None:
    with tempfile.NamedTemporaryFile() as file:
        file_storage = optuna.storages.JournalFileStorage(file.name)
        storage = optuna.storages.JournalStorage(file_storage)
        study = optuna.create_study(storage=storage)
        num_enqueued = 10
        for i in range(num_enqueued):
            study.enqueue_trial({"i": i})

        trial_id_set = set()
        with ProcessPoolExecutor(10) as pool:
            futures = []
            for i in range(num_enqueued):
                future = pool.submit(pop_waiting_trial, file.name, study.study_name)
                futures.append(future)

            for future in as_completed(futures):
                trial_id = future.result()
                if trial_id is not None:
                    trial_id_set.add(trial_id)
        assert len(trial_id_set) == num_enqueued


@pytest.mark.parametrize("log_storage_type", LOG_STORAGE)
def test_pickle_dump_and_load(log_storage_type: str) -> None:
    if log_storage_type.startswith("redis"):
        pytest.skip("The `JournalRedisStorage` is not pickalable.")

    with JournalLogStorageSupplier(log_storage_type) as file_storage:
        storage = optuna.storages.JournalStorage(file_storage)
        study = optuna.create_study(storage=storage)
        num_enqueued = 10
        for i in range(num_enqueued):
            study.enqueue_trial({"i": i})
        loaded_storage = pickle.loads(pickle.dumps(storage))
        study_id = loaded_storage.get_study_id_from_name(study.study_name)
        loaded_trials = loaded_storage.get_all_trials(study_id=study_id)
        assert len(loaded_trials) == num_enqueued
        trials = storage.get_all_trials(study_id=study_id)
        for trial, loaded_trial in zip(trials, loaded_trials):
            assert trial == loaded_trial
