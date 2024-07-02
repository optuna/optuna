from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import pickle
from types import TracebackType
from typing import Any
from typing import IO
from typing import Optional
from typing import Type
from unittest import mock

import _pytest.capture
from fakeredis import FakeStrictRedis
import pytest

import optuna
from optuna import create_study
from optuna.storages import JournalStorage
from optuna.storages._journal.base import BaseJournalLogSnapshot
from optuna.storages._journal.file import JournalFileBaseLock
from optuna.storages._journal.storage import JournalStorageReplayResult
from optuna.testing.storages import StorageSupplier
from optuna.testing.tempfile_pool import NamedTemporaryFilePool


LOG_STORAGE = [
    "file_with_open_lock",
    "file_with_link_lock",
    "redis_default",
    "redis_with_use_cluster",
]

JOURNAL_STORAGE_SUPPORTING_SNAPSHOT = ["journal_redis"]


class JournalLogStorageSupplier:
    def __init__(self, storage_type: str, deprecated_classname: bool = False) -> None:
        self.storage_type = storage_type
        self.tempfile: Optional[IO[Any]] = None
        self.deprecated_classname = deprecated_classname

    def __enter__(self) -> optuna.storages.BaseJournalLogStorage:
        if self.storage_type.startswith("file"):
            self.tempfile = NamedTemporaryFilePool().tempfile()
            lock: JournalFileBaseLock
            if self.storage_type == "file_with_open_lock":
                lock = optuna.storages.JournalFileOpenLock(self.tempfile.name)
            elif self.storage_type == "file_with_link_lock":
                lock = optuna.storages.JournalFileSymlinkLock(self.tempfile.name)
            else:
                raise Exception("Must not reach here")
            if self.deprecated_classname:
                return optuna.storages.JournalFileStorage(self.tempfile.name, lock)
            else:
                return optuna.storages.JournalFileBackend(self.tempfile.name, lock)
        elif self.storage_type.startswith("redis"):
            use_cluster = self.storage_type == "redis_with_use_cluster"
            if self.deprecated_classname:
                journal_redis_storage = optuna.storages.JournalRedisStorage(
                    "redis://localhost", use_cluster
                )
            else:
                journal_redis_storage = optuna.storages.JournalRedisBackend(
                    "redis://localhost", use_cluster
                )
            journal_redis_storage._redis = FakeStrictRedis()  # type: ignore[no-untyped-call]
            return journal_redis_storage
        else:
            raise RuntimeError("Unknown log storage type: {}".format(self.storage_type))

    def __exit__(
        self, exc_type: Type[BaseException], exc_val: BaseException, exc_tb: TracebackType
    ) -> None:
        if self.tempfile:
            self.tempfile.close()


@pytest.mark.parametrize("log_storage_type", LOG_STORAGE)
@pytest.mark.parametrize("deprecated_classname", [True, False])
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_concurrent_append_logs_for_multi_processes(
    log_storage_type: str, deprecated_classname: bool
) -> None:
    if log_storage_type.startswith("redis"):
        pytest.skip("The `fakeredis` does not support multi process environments.")

    num_executors = 10
    num_records = 200
    record = {"key": "value"}

    with JournalLogStorageSupplier(log_storage_type, deprecated_classname) as storage:
        with ProcessPoolExecutor(num_executors) as pool:
            pool.map(storage.append_logs, [[record] for _ in range(num_records)], timeout=20)

        assert len(storage.read_logs(0)) == num_records
        assert all(record == r for r in storage.read_logs(0))


@pytest.mark.parametrize("log_storage_type", LOG_STORAGE)
@pytest.mark.parametrize("deprecated_classname", [True, False])
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_concurrent_append_logs_for_multi_threads(
    log_storage_type: str, deprecated_classname: bool
) -> None:
    num_executors = 10
    num_records = 200
    record = {"key": "value"}

    with JournalLogStorageSupplier(log_storage_type, deprecated_classname) as storage:
        with ThreadPoolExecutor(num_executors) as pool:
            pool.map(storage.append_logs, [[record] for _ in range(num_records)], timeout=20)

        assert len(storage.read_logs(0)) == num_records
        assert all(record == r for r in storage.read_logs(0))


def pop_waiting_trial(
    file_path: str, study_name: str, deprecated_classname: bool
) -> Optional[int]:
    if deprecated_classname:
        file_storage = optuna.storages.JournalFileStorage(file_path)
    else:
        file_storage = optuna.storages.JournalFileBackend(file_path)
    storage = optuna.storages.JournalStorage(file_storage)
    study = optuna.load_study(storage=storage, study_name=study_name)
    return study._pop_waiting_trial_id()


@pytest.mark.parametrize("deprecated_classname", [True, False])
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_pop_waiting_trial_multiprocess_safe(deprecated_classname: bool) -> None:
    with NamedTemporaryFilePool() as file:
        if deprecated_classname:
            file_storage = optuna.storages.JournalFileStorage(file.name)
        else:
            file_storage = optuna.storages.JournalFileBackend(file.name)
        storage = optuna.storages.JournalStorage(file_storage)
        study = optuna.create_study(storage=storage)
        num_enqueued = 10
        for i in range(num_enqueued):
            study.enqueue_trial({"i": i})

        trial_id_set = set()
        with ProcessPoolExecutor(10) as pool:
            futures = []
            for i in range(num_enqueued):
                future = pool.submit(
                    pop_waiting_trial, file.name, study.study_name, deprecated_classname
                )
                futures.append(future)

            for future in as_completed(futures):
                trial_id = future.result()
                if trial_id is not None:
                    trial_id_set.add(trial_id)
        assert len(trial_id_set) == num_enqueued


@pytest.mark.parametrize("storage_mode", JOURNAL_STORAGE_SUPPORTING_SNAPSHOT)
def test_save_snapshot_per_each_trial(storage_mode: str) -> None:
    def objective(trial: optuna.Trial) -> float:
        return trial.suggest_float("x", 0, 10)

    with StorageSupplier(storage_mode) as storage:
        assert isinstance(storage, JournalStorage)
        study = create_study(storage=storage)
        journal_log_storage = storage._backend
        assert isinstance(journal_log_storage, BaseJournalLogSnapshot)

        assert journal_log_storage.load_snapshot() is None

        with mock.patch("optuna.storages._journal.storage.SNAPSHOT_INTERVAL", 1, create=True):
            study.optimize(objective, n_trials=2)

        assert isinstance(journal_log_storage.load_snapshot(), bytes)


@pytest.mark.parametrize("storage_mode", JOURNAL_STORAGE_SUPPORTING_SNAPSHOT)
def test_save_snapshot_per_each_study(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        assert isinstance(storage, JournalStorage)
        journal_log_storage = storage._backend
        assert isinstance(journal_log_storage, BaseJournalLogSnapshot)

        assert journal_log_storage.load_snapshot() is None

        with mock.patch("optuna.storages._journal.storage.SNAPSHOT_INTERVAL", 1, create=True):
            for _ in range(2):
                create_study(storage=storage)

        assert isinstance(journal_log_storage.load_snapshot(), bytes)


@pytest.mark.parametrize("storage_mode", JOURNAL_STORAGE_SUPPORTING_SNAPSHOT)
def test_check_replay_result_restored_from_snapshot(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage1:
        with mock.patch("optuna.storages._journal.storage.SNAPSHOT_INTERVAL", 1, create=True):
            for _ in range(2):
                create_study(storage=storage1)

        assert isinstance(storage1, JournalStorage)
        storage2 = optuna.storages.JournalStorage(storage1._backend)
        assert len(storage1.get_all_studies()) == len(storage2.get_all_studies())
        assert storage1._replay_result.log_number_read == storage2._replay_result.log_number_read


@pytest.mark.parametrize("storage_mode", JOURNAL_STORAGE_SUPPORTING_SNAPSHOT)
def test_snapshot_given(storage_mode: str, capsys: _pytest.capture.CaptureFixture) -> None:
    with StorageSupplier(storage_mode) as storage:
        assert isinstance(storage, JournalStorage)
        replay_result = JournalStorageReplayResult("")
        # Bytes object which is a valid pickled object.
        storage.restore_replay_result(pickle.dumps(replay_result))
        assert replay_result.log_number_read == storage._replay_result.log_number_read

        # We need to reconstruct our default handler to properly capture stderr.
        optuna.logging._reset_library_root_logger()
        optuna.logging.enable_default_handler()
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Bytes object which cannot be unpickled is passed.
        storage.restore_replay_result(b"hoge")
        _, err = capsys.readouterr()
        assert err

        # Bytes object which can be pickled but is not `JournalStorageReplayResult`.
        storage.restore_replay_result(pickle.dumps("hoge"))
        _, err = capsys.readouterr()
        assert err


def test_if_future_warning_occurs() -> None:
    with NamedTemporaryFilePool() as file:
        with pytest.warns(FutureWarning):
            optuna.storages.JournalFileStorage(file.name)

    with pytest.warns(FutureWarning):
        optuna.storages.JournalRedisStorage("redis://localhost")
