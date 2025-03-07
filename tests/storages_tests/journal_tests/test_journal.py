from __future__ import annotations

from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import pathlib
import pickle
from types import TracebackType
from typing import Any
from typing import IO
from unittest import mock

import _pytest.capture
from fakeredis import FakeStrictRedis
import pytest

import optuna
from optuna import create_study
from optuna.storages import BaseJournalLogStorage
from optuna.storages import journal
from optuna.storages import JournalFileOpenLock as DeprecatedJournalFileOpenLock
from optuna.storages import JournalFileSymlinkLock as DeprecatedJournalFileSymlinkLock
from optuna.storages import JournalStorage
from optuna.storages.journal._base import BaseJournalSnapshot
from optuna.storages.journal._file import BaseJournalFileLock
from optuna.storages.journal._storage import JournalStorageReplayResult
from optuna.testing.storages import StorageSupplier
from optuna.testing.tempfile_pool import NamedTemporaryFilePool


LOG_STORAGE_WITH_PARAMETER = [
    ("file_with_open_lock", 30),
    ("file_with_open_lock", None),
    ("file_with_link_lock", 30),
    ("file_with_link_lock", None),
    ("redis_default", None),
    ("redis_with_use_cluster", None),
]

JOURNAL_STORAGE_SUPPORTING_SNAPSHOT = ["journal_redis"]


class JournalLogStorageSupplier:
    def __init__(self, storage_type: str, grace_period: int | None) -> None:
        self.storage_type = storage_type
        self.tempfile: IO[Any] | None = None
        self.grace_period = grace_period

    def __enter__(self) -> optuna.storages.journal.BaseJournalBackend:
        if self.storage_type.startswith("file"):
            self.tempfile = NamedTemporaryFilePool().tempfile()
            lock: BaseJournalFileLock
            if self.storage_type == "file_with_open_lock":
                lock = optuna.storages.journal.JournalFileOpenLock(
                    self.tempfile.name, self.grace_period
                )
            elif self.storage_type == "file_with_link_lock":
                lock = optuna.storages.journal.JournalFileOpenLock(
                    self.tempfile.name, self.grace_period
                )
            else:
                raise Exception("Must not reach here")
            return optuna.storages.journal.JournalFileBackend(self.tempfile.name, lock)
        elif self.storage_type.startswith("redis"):
            assert self.grace_period is None
            use_cluster = self.storage_type == "redis_with_use_cluster"
            journal_redis_storage = optuna.storages.journal.JournalRedisBackend(
                "redis://localhost", use_cluster
            )
            journal_redis_storage._redis = FakeStrictRedis()  # type: ignore[no-untyped-call]
            return journal_redis_storage
        else:
            raise RuntimeError("Unknown log storage type: {}".format(self.storage_type))

    def __exit__(
        self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType
    ) -> None:
        if self.tempfile:
            self.tempfile.close()


@pytest.mark.parametrize("log_storage_type,grace_period", LOG_STORAGE_WITH_PARAMETER)
def test_concurrent_append_logs_for_multi_processes(
    log_storage_type: str, grace_period: int | None
) -> None:
    if log_storage_type.startswith("redis"):
        pytest.skip("The `fakeredis` does not support multi process environments.")

    num_executors = 10
    num_records = 200
    record = {"key": "value"}

    with JournalLogStorageSupplier(log_storage_type, grace_period) as storage:
        with ProcessPoolExecutor(num_executors) as pool:
            pool.map(storage.append_logs, [[record] for _ in range(num_records)], timeout=20)

        assert len(storage.read_logs(0)) == num_records
        assert all(record == r for r in storage.read_logs(0))


@pytest.mark.parametrize("log_storage_type,grace_period", LOG_STORAGE_WITH_PARAMETER)
def test_concurrent_append_logs_for_multi_threads(
    log_storage_type: str, grace_period: int | None
) -> None:
    num_executors = 10
    num_records = 200
    record = {"key": "value"}

    with JournalLogStorageSupplier(log_storage_type, grace_period) as storage:
        with ThreadPoolExecutor(num_executors) as pool:
            pool.map(storage.append_logs, [[record] for _ in range(num_records)], timeout=20)

        assert len(storage.read_logs(0)) == num_records
        assert all(record == r for r in storage.read_logs(0))


def pop_waiting_trial(file_path: str, study_name: str) -> int | None:
    file_storage = optuna.storages.journal.JournalFileBackend(file_path)
    storage = optuna.storages.JournalStorage(file_storage)
    study = optuna.load_study(storage=storage, study_name=study_name)
    return study._pop_waiting_trial_id()


def test_pop_waiting_trial_multiprocess_safe() -> None:
    with NamedTemporaryFilePool() as file:
        file_storage = optuna.storages.journal.JournalFileBackend(file.name)
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


@pytest.mark.parametrize("storage_mode", JOURNAL_STORAGE_SUPPORTING_SNAPSHOT)
def test_save_snapshot_per_each_trial(storage_mode: str) -> None:
    def objective(trial: optuna.Trial) -> float:
        return trial.suggest_float("x", 0, 10)

    with StorageSupplier(storage_mode) as storage:
        assert isinstance(storage, JournalStorage)
        study = create_study(storage=storage)
        journal_log_storage = storage._backend
        assert isinstance(journal_log_storage, BaseJournalSnapshot)

        assert journal_log_storage.load_snapshot() is None

        with mock.patch("optuna.storages.journal._storage.SNAPSHOT_INTERVAL", 1, create=True):
            study.optimize(objective, n_trials=2)

        assert isinstance(journal_log_storage.load_snapshot(), bytes)


@pytest.mark.parametrize("storage_mode", JOURNAL_STORAGE_SUPPORTING_SNAPSHOT)
def test_save_snapshot_per_each_study(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        assert isinstance(storage, JournalStorage)
        journal_log_storage = storage._backend
        assert isinstance(journal_log_storage, BaseJournalSnapshot)

        assert journal_log_storage.load_snapshot() is None

        with mock.patch("optuna.storages.journal._storage.SNAPSHOT_INTERVAL", 1, create=True):
            for _ in range(2):
                create_study(storage=storage)

        assert isinstance(journal_log_storage.load_snapshot(), bytes)


@pytest.mark.parametrize("storage_mode", JOURNAL_STORAGE_SUPPORTING_SNAPSHOT)
def test_check_replay_result_restored_from_snapshot(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage1:
        with mock.patch("optuna.storages.journal._storage.SNAPSHOT_INTERVAL", 1, create=True):
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

    class _CustomJournalBackendInheritingDeprecatedClass(BaseJournalLogStorage):
        def read_logs(self, log_number_from: int) -> list[dict[str, Any]]:
            return [{"": ""}]

        def append_logs(self, logs: list[dict[str, Any]]) -> None:
            return

    with pytest.warns(FutureWarning):
        _ = _CustomJournalBackendInheritingDeprecatedClass()


@pytest.mark.parametrize(
    "lock_obj", (DeprecatedJournalFileOpenLock, DeprecatedJournalFileSymlinkLock)
)
def test_future_warning_of_deprecated_file_lock_obj_paths(
    tmp_path: pathlib.PurePath,
    lock_obj: type[DeprecatedJournalFileOpenLock | DeprecatedJournalFileSymlinkLock],
) -> None:
    with pytest.warns(FutureWarning):
        lock_obj(filepath=str(tmp_path))


def test_raise_error_for_deprecated_class_import_from_journal() -> None:
    # TODO(nabenabe0928): Remove this test once deprecated objects, e.g., JournalFileStorage,
    # are removed.
    with pytest.raises(AttributeError):
        journal.JournalFileStorage  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        journal.JournalRedisStorage  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        journal.BaseJournalLogStorage  # type: ignore[attr-defined]


@pytest.mark.parametrize("log_storage_type", ("file_with_open_lock", "file_with_link_lock"))
@pytest.mark.parametrize("grace_period", (0, -1))
def test_invalid_grace_period(log_storage_type: str, grace_period: int) -> None:
    with pytest.raises(ValueError):
        with JournalLogStorageSupplier(log_storage_type, grace_period):
            pass
