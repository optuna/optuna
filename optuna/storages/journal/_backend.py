from __future__ import annotations

import json
import os
from typing import Any

from optuna._deprecated import deprecated_class
from optuna.storages.journal._base import BaseJournalBackend
from optuna.storages.journal._base import BaseJournalFileLock
from optuna.storages.journal._file_lock import get_lock_file
from optuna.storages.journal._file_lock import JournalFileSymlinkLock


class JournalFileBackend(BaseJournalBackend):
    """File storage class for Journal log backend.

    Compared to SQLite3, the benefit of this backend is that it is more suitable for
    environments where the file system does not support ``fcntl()`` file locking.
    For example, as written in the `SQLite3 FAQ <https://www.sqlite.org/faq.html#q5>`__,
    SQLite3 might not work on NFS (Network File System) since ``fcntl()`` file locking
    is broken on many NFS implementations. In such scenarios, this backend provides
    several workarounds for locking files. For more details, refer to the `Medium blog post`_.

    .. _Medium blog post: https://medium.com/optuna/distributed-optimization-via-nfs\
    -using-optunas-new-operation-based-logging-storage-9815f9c3f932

    It's important to note that, similar to SQLite3, this class doesn't support a high
    level of write concurrency, as outlined in the `SQLAlchemy documentation`_. However,
    in typical situations where the objective function is computationally expensive, Optuna
    users don't need to be concerned about this limitation. The reason being, the write
    operations are not the bottleneck as long as the objective function doesn't invoke
    :meth:`~optuna.trial.Trial.report` and :meth:`~optuna.trial.Trial.set_user_attr` excessively.

    .. _SQLAlchemy documentation: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html\
    #database-locking-behavior-concurrency

    Args:
        file_path:
            Path of file to persist the log to.

        lock_obj:
            Lock object for process exclusivity. An instance of
            :class:`~optuna.storages.JournalFileSymlinkLock` and
            :class:`~optuna.storages.JournalFileOpenLock` can be passed.
    """

    def __init__(self, file_path: str, lock_obj: BaseJournalFileLock | None = None) -> None:
        self._file_path: str = file_path
        self._lock = lock_obj or JournalFileSymlinkLock(self._file_path)
        if not os.path.exists(self._file_path):
            open(self._file_path, "ab").close()  # Create a file if it does not exist.
        self._log_number_offset: dict[int, int] = {0: 0}

    def read_logs(self, log_number_from: int) -> list[dict[str, Any]]:
        logs = []
        with open(self._file_path, "rb") as f:
            # Maintain remaining_log_size to allow writing by another process
            # while reading the log.
            remaining_log_size = os.stat(self._file_path).st_size
            log_number_start = 0
            if log_number_from in self._log_number_offset:
                f.seek(self._log_number_offset[log_number_from])
                log_number_start = log_number_from
                remaining_log_size -= self._log_number_offset[log_number_from]

            last_decode_error = None
            for log_number, line in enumerate(f, start=log_number_start):
                byte_len = len(line)
                remaining_log_size -= byte_len
                if remaining_log_size < 0:
                    break
                if last_decode_error is not None:
                    raise last_decode_error
                if log_number + 1 not in self._log_number_offset:
                    self._log_number_offset[log_number + 1] = (
                        self._log_number_offset[log_number] + byte_len
                    )
                if log_number < log_number_from:
                    continue

                # Ensure that each line ends with line separators (\n, \r\n).
                if not line.endswith(b"\n"):
                    last_decode_error = ValueError("Invalid log format.")
                    del self._log_number_offset[log_number + 1]
                    continue
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError as err:
                    last_decode_error = err
                    del self._log_number_offset[log_number + 1]
            return logs

    def append_logs(self, logs: list[dict[str, Any]]) -> None:
        with get_lock_file(self._lock):
            what_to_write = (
                "\n".join([json.dumps(log, separators=(",", ":")) for log in logs]) + "\n"
            )
            with open(self._file_path, "ab") as f:
                f.write(what_to_write.encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())


@deprecated_class(
    "4.0.0", "7.0.0", text="Use :class:`~optuna.storages.JournalFileBackend` instead."
)
class JournalFileStorage(JournalFileBackend):
    """File storage class for Journal log backend.

    Compared to SQLite3, the benefit of this backend is that it is more suitable for
    environments where the file system does not support ``fcntl()`` file locking.
    For example, as written in the `SQLite3 FAQ <https://www.sqlite.org/faq.html#q5>`__,
    SQLite3 might not work on NFS (Network File System) since ``fcntl()`` file locking
    is broken on many NFS implementations. In such scenarios, this backend provides
    several workarounds for locking files. For more details, refer to the `Medium blog post`_.

    .. _Medium blog post: https://medium.com/optuna/distributed-optimization-via-nfs\
    -using-optunas-new-operation-based-logging-storage-9815f9c3f932

    It's important to note that, similar to SQLite3, this class doesn't support a high
    level of write concurrency, as outlined in the `SQLAlchemy documentation`_. However,
    in typical situations where the objective function is computationally expensive, Optuna
    users don't need to be concerned about this limitation. The reason being, the write
    operations are not the bottleneck as long as the objective function doesn't invoke
    :meth:`~optuna.trial.Trial.report` and :meth:`~optuna.trial.Trial.set_user_attr` excessively.

    .. _SQLAlchemy documentation: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html\
    #database-locking-behavior-concurrency

    Args:
        file_path:
            Path of file to persist the log to.

        lock_obj:
            Lock object for process exclusivity.
    """

    def __init__(self, file_path: str, lock_obj: BaseJournalFileLock | None = None) -> None:
        super().__init__(file_path=file_path, lock_obj=lock_obj)
