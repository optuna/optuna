import abc
from contextlib import contextmanager
import errno
import json
import os
import time
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
import uuid

from optuna.storages._journal.base import BaseJournalLogStorage


LOCK_FILE_SUFFIX = ".lock"
RENAME_FILE_SUFFIX = ".rename"


class JournalFileBaseLock(abc.ABC):
    @abc.abstractmethod
    def acquire(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def release(self) -> None:
        raise NotImplementedError


class JournalFileSymlinkLock(JournalFileBaseLock):
    """Lock class for synchronizing processes for NFSv2 or later.

    On acquiring the lock, link system call is called to create an exclusive file. The file is
    deleted when the lock is released. In NFS environments prior to NFSv3, use this instead of
    :class:`~optuna.storages.JournalFileOpenLock`

    Args:
        filepath:
            The path of the file whose race condition must be protected.
    """

    def __init__(self, filepath: str) -> None:
        self._lock_target_file = filepath
        self._lock_file = filepath + LOCK_FILE_SUFFIX
        self._lock_rename_file = self._lock_file + str(uuid.uuid4()) + RENAME_FILE_SUFFIX

    def acquire(self) -> bool:
        """Acquire a lock in a blocking way by creating a symbolic link of a file.

        Returns:
            :obj:`True` if it succeeded in creating a symbolic link of `self._lock_target_file`.

        """
        sleep_secs = 0.001
        while True:
            try:
                os.symlink(self._lock_target_file, self._lock_file)
                return True
            except OSError as err:
                if err.errno == errno.EEXIST:
                    time.sleep(sleep_secs)
                    sleep_secs = min(sleep_secs * 2, 1)
                    continue
                raise err
            except BaseException:
                self.release()
                raise

    def release(self) -> None:
        """Release a lock by removing the symbolic link."""

        try:
            os.rename(self._lock_file, self._lock_rename_file)
            os.unlink(self._lock_rename_file)
        except OSError:
            raise RuntimeError("Error: did not possess lock")
        except BaseException:
            os.unlink(self._lock_rename_file)
            raise


class JournalFileOpenLock(JournalFileBaseLock):
    """Lock class for synchronizing processes for NFSv3 or later.

    On acquiring the lock, open system call is called with the O_EXCL option to create an exclusive
    file. The file is deleted when the lock is released. This class is only supported when using
    NFSv3 or later on kernel 2.6 or later. In prior NFS environments, use
    :class:`~optuna.storages.JournalFileSymlinkLock`.

    Args:
        filepath:
            The path of the file whose race condition must be protected.
    """

    def __init__(self, filepath: str) -> None:
        self._lock_file = filepath + LOCK_FILE_SUFFIX
        self._lock_rename_file = self._lock_file + str(uuid.uuid4()) + RENAME_FILE_SUFFIX

    def acquire(self) -> bool:
        """Acquire a lock in a blocking way by creating a lock file.

        Returns:
            :obj:`True` if it succeeded in creating a `self._lock_file`

        """
        sleep_secs = 0.001
        while True:
            try:
                open_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                os.close(os.open(self._lock_file, open_flags))
                return True
            except OSError as err:
                if err.errno == errno.EEXIST:
                    time.sleep(sleep_secs)
                    sleep_secs = min(sleep_secs * 2, 1)
                    continue
                raise err
            except BaseException:
                self.release()
                raise

    def release(self) -> None:
        """Release a lock by removing the created file."""

        try:
            os.rename(self._lock_file, self._lock_rename_file)
            os.unlink(self._lock_rename_file)
        except OSError:
            raise RuntimeError("Error: did not possess lock")
        except BaseException:
            os.unlink(self._lock_rename_file)
            raise


@contextmanager
def get_lock_file(lock_obj: JournalFileBaseLock) -> Iterator[None]:
    lock_obj.acquire()
    try:
        yield
    finally:
        lock_obj.release()


class JournalFileStorage(BaseJournalLogStorage):
    """File storage class for Journal log backend.

    Args:
        file_path:
            Path of file to persist the log to.

        lock_obj:
            Lock object for process exclusivity.

    """

    def __init__(self, file_path: str, lock_obj: Optional[JournalFileBaseLock] = None) -> None:
        self._file_path: str = file_path
        self._lock = lock_obj or JournalFileSymlinkLock(self._file_path)
        open(self._file_path, "ab").close()  # Create a file if it does not exist
        self._log_number_offset: Dict[int, int] = {0: 0}

    def read_logs(self, log_number_from: int) -> List[Dict[str, Any]]:
        logs = []
        with open(self._file_path, "rb") as f:
            log_number_start = 0
            if log_number_from in self._log_number_offset:
                f.seek(self._log_number_offset[log_number_from])
                log_number_start = log_number_from

            last_decode_error = None
            for log_number, line in enumerate(f, start=log_number_start):
                if last_decode_error is not None:
                    raise last_decode_error
                if log_number + 1 not in self._log_number_offset:
                    byte_len = len(line)
                    self._log_number_offset[log_number + 1] = (
                        self._log_number_offset[log_number] + byte_len
                    )
                if log_number < log_number_from:
                    continue
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError as err:
                    last_decode_error = err
                    del self._log_number_offset[log_number + 1]

            return logs

    def append_logs(self, logs: List[Dict[str, Any]]) -> None:
        with get_lock_file(self._lock):
            what_to_write = "\n".join([json.dumps(log) for log in logs]) + "\n"
            with open(self._file_path, "ab") as f:
                f.write(what_to_write.encode("utf-8"))
                f.flush()
                os.fsync(f.fileno())
