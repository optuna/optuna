from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import tempfile
from types import TracebackType
from typing import Any
from typing import IO
from typing import Optional
from typing import Type

import pytest

import optuna
from optuna.storages._journal.file import JournalFileLinkLock
from optuna.storages._journal.file import JournalFileLockType
from optuna.storages._journal.file import JournalFileOpenLock


LOG_STORAGE = {
    "file_with_open_lock",
    "file_with_link_lock",
}


class JournalLogStorageSupplier:
    def __init__(self, storage_type: str) -> None:
        self.storage_type = storage_type
        self.tempfile: Optional[IO[Any]] = None

    def __enter__(self) -> optuna.storages.JournalFileStorage:
        if self.storage_type.startswith("file"):
            self.tempfile = tempfile.NamedTemporaryFile()
            lock: JournalFileLockType
            if self.storage_type == "file_with_open_lock":
                lock = JournalFileOpenLock(self.tempfile.name)
            elif self.storage_type == "file_with_link_lock":
                lock = JournalFileLinkLock(self.tempfile.name)
            else:
                raise Exception("Must not reach here")
            return optuna.storages.JournalFileStorage(self.tempfile.name, lock)
        else:
            raise RuntimeError("Unknown log storage type: {}".format(self.storage_type))

    def __exit__(
        self, exc_type: Type[BaseException], exc_val: BaseException, exc_tb: TracebackType
    ) -> None:

        if self.tempfile:
            self.tempfile.close()


@pytest.mark.parametrize("log_storage_type", LOG_STORAGE)
def test_concurrent_append_logs(log_storage_type: str) -> None:
    num_executors = 10
    num_records = 200
    record = {"key": "value"}

    with JournalLogStorageSupplier(log_storage_type) as storage:
        with ProcessPoolExecutor(num_executors) as pool:
            pool.map(storage.append_logs, [[record] for _ in range(num_records)], timeout=20)

        assert len(storage.get_unread_logs(0)) == num_records
        assert all(record == r for r in storage.get_unread_logs(0))

    with JournalLogStorageSupplier(log_storage_type) as storage:
        with ThreadPoolExecutor(num_executors) as pool:
            pool.map(storage.append_logs, [[record] for _ in range(num_records)], timeout=20)

        assert len(storage.get_unread_logs(0)) == num_records
        assert all(record == r for r in storage.get_unread_logs(0))
