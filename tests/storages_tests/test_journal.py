import pytest
import tempfile
import optuna
from typing import Optional, IO, Any, Type

from optuna.storages._journal.file import BaseFileLock, OpenLock, LinkLock
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from types import TracebackType


LOCK_TYPE = {
    "open_lock",
    "link_lock",
}

LOG_STORAGE = {
    "file",
}


class JournalLogStorageSupplier:
    def __init__(self, storage_type: str, **kwargs: Any) -> None:
        self.storage_type = storage_type
        self.tempfile: Optional[IO[Any]] = None
        self.extra_args = kwargs

    def _get_lock(self, lock_type: str, lock_file_path: str) -> BaseFileLock:
        if lock_type == "open_lock":
            return OpenLock(lock_file_path)
        elif lock_type == "link_lock":
            return LinkLock(lock_file_path)
        else:
            raise RuntimeError("Unknown lock type: {}".format(lock_type))

    def __enter__(self) -> optuna.storages.JournalFileStorage:
        if self.storage_type == "file":
            self.tempfile = tempfile.NamedTemporaryFile()
            lock = self._get_lock(self.extra_args.get("lock", "open_lock"), self.tempfile.name)
            return optuna.storages.JournalFileStorage(self.tempfile.name, lock)
        else:
            raise RuntimeError("Unknown log storage type: {}".format(self.storage_type))

    def __exit__(
        self, exc_type: Type[BaseException], exc_val: BaseException, exc_tb: TracebackType
    ) -> None:

        if self.tempfile:
            self.tempfile.close()


@pytest.mark.parametrize("lock_type", LOCK_TYPE)
@pytest.mark.parametrize("log_storage_type", LOG_STORAGE)
def test_concurrent_append_logs(lock_type: str, log_storage_type: str) -> None:
    num_executors = 10
    num_records = 200
    record = {"key": "value"}

    with JournalLogStorageSupplier(log_storage_type, lock=lock_type) as storage:
        with ProcessPoolExecutor(num_executors) as pool:
            pool.map(storage.append_logs, [[record] for _ in range(num_records)], timeout=20)

        assert len(storage.get_unread_logs(0)) == num_records
        assert all(record == r for r in storage.get_unread_logs(0))

    with JournalLogStorageSupplier(log_storage_type, lock=lock_type) as storage:
        with ThreadPoolExecutor(num_executors) as pool:
            pool.map(storage.append_logs, [[record] for _ in range(num_records)], timeout=20)

        assert len(storage.get_unread_logs(0)) == num_records
        assert all(record == r for r in storage.get_unread_logs(0))


@pytest.mark.parametrize("lock_type", LOCK_TYPE)
@pytest.mark.parametrize("log_storage_type", LOG_STORAGE)
def test_get_unread_logs(lock_type: str, log_storage_type: str):
    num_records = 30
    record = {"key": "value"}

    with JournalLogStorageSupplier(log_storage_type, lock=lock_type) as storage:
        storage.append_logs([record for _ in range(num_records)])
        assert all(len(storage.get_unread_logs(i)) == num_records - i for i in range(num_records))
