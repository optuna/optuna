from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import pickle
import tempfile
from types import TracebackType
from typing import Any
from typing import Dict
from typing import IO
from typing import Optional
from typing import Type
from unittest import mock

import fakeredis
import pytest

import optuna
from optuna.storages._journal.base import SnapshotRestoreError
from optuna.storages._journal.file import JournalFileBaseLock
from optuna.testing.storages import StorageSupplier


LOG_STORAGE = {
    "file_with_open_lock",
    "file_with_link_lock",
    "redis_default",
    "redis_with_use_cluster",
}

LOG_STORAGE_SUPPORTING_SNAPSHOT = {
    "redis_default",
    "redis_with_use_cluster",
}

JOURNAL_STORAGE_SUPPORTING_SNAPSHOT = {"journal_redis"}


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


@pytest.mark.parametrize("log_storage_type", LOG_STORAGE_SUPPORTING_SNAPSHOT)
def test_load_snapshot(log_storage_type: str) -> None:
    with JournalLogStorageSupplier(log_storage_type) as storage:
        assert isinstance(storage, optuna.storages.JournalRedisStorage)

        class Loader(object):
            def __init__(self) -> None:
                self.n_called = 0

            def __call__(self, snapshot: bytes) -> None:
                self.n_called += 1
                raise SnapshotRestoreError

        loader = Loader()

        # Any snapshots are now saved, so `load_snapshot` is early returned.
        storage.load_snapshot(loader)
        assert loader.n_called == 0

        storage.save_snapshot(pickle.dumps("dummy_snapshot"))
        # After even one snapshot has been saved, the `loader` should be called.
        storage.load_snapshot(loader)
        assert loader.n_called == 1


@pytest.mark.parametrize(
    "storage_mode, kwargs",
    zip(JOURNAL_STORAGE_SUPPORTING_SNAPSHOT, [{"redis": fakeredis.FakeStrictRedis()}]),
)
def test_snapshot(storage_mode: str, kwargs: Dict[str, Any]) -> None:
    with mock.patch("optuna.storages._journal.storage.SNAPSHOT_INTERVAL", 1, create=True):
        with StorageSupplier(storage_mode, **kwargs) as storage1:
            assert isinstance(storage1, optuna.storages.JournalStorage)

            study_id = storage1.create_new_study()

            # The snapshot is not saved here.
            storage1.create_new_trial(study_id)

            # The number of read logs are same thanks to `sync_with_backend` without any snapshots.
            with StorageSupplier(storage_mode, **kwargs) as storage2:
                assert isinstance(storage2, optuna.storages.JournalStorage)
                assert (
                    storage1._replay_result.log_number_read
                    == storage2._replay_result.log_number_read
                )

            # The number of read logs are not same since `sync_with_backend` is disabled and the
            # snapshot was not saved.
            with mock.patch(
                "optuna.storages._journal.storage.JournalStorage._sync_with_backend"
            ) as m:
                with StorageSupplier(storage_mode, **kwargs) as storage2:
                    assert isinstance(storage2, optuna.storages.JournalStorage)
                    m.assert_called_once()
                    assert (
                        storage1._replay_result.log_number_read
                        != storage2._replay_result.log_number_read
                    )

            # The snapshot is saved here.
            storage1.create_new_trial(study_id)

            # The number of read logs are same thanks to the snapshot without `sync_with_backend`.
            with mock.patch(
                "optuna.storages._journal.storage.JournalStorage._sync_with_backend"
            ) as m:
                with StorageSupplier(storage_mode, **kwargs) as storage2:
                    assert isinstance(storage2, optuna.storages.JournalStorage)
                    m.assert_called_once()
                    assert (
                        storage1._replay_result.log_number_read
                        == storage2._replay_result.log_number_read
                    )

            # The snapshot is saved here.
            _ = storage1.create_new_study()

            # The number of read logs are same thanks to the snapshot without `sync_with_backend`.
            with mock.patch(
                "optuna.storages._journal.storage.JournalStorage._sync_with_backend"
            ) as m:
                with StorageSupplier(storage_mode, **kwargs) as storage2:
                    assert isinstance(storage2, optuna.storages.JournalStorage)
                    m.assert_called_once()
                    assert (
                        storage1._replay_result.log_number_read
                        == storage2._replay_result.log_number_read
                    )


@pytest.mark.parametrize("storage_mode", JOURNAL_STORAGE_SUPPORTING_SNAPSHOT)
def test_invalid_snapshot(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        assert isinstance(storage, optuna.storages.JournalStorage)
        # Bytes object which cannot be unpickled is passed.
        with pytest.raises(SnapshotRestoreError):
            storage.restore_replay_result(b"hoge")

        # Bytes object which can be pickeled but is not `JournalStorageReplayResult`.
        with pytest.raises(SnapshotRestoreError):
            storage.restore_replay_result(pickle.dumps("hoge"))
