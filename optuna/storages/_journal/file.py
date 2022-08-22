import abc
from contextlib import contextmanager
import errno
import json
import os
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
import uuid

from optuna._experimental import experimental_class
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


@experimental_class("3.1.0")
class JournalFileSymlinkLock(JournalFileBaseLock):
    """Lock class for synchronizing processes.

    On acquiring the lock, link system call is called to create an exclusive file. The file is
    deleted when the lock is released. In NFS environments prior to NFSv3, use this instead of
    `~optuna.storages.JournalFileOpenLock`

    Args:
        filepath:
            The path of the file whose race condition must be protected.
    """

    def __init__(self, filepath: str) -> None:
        self._lock_target_file = filepath
        self._lockfile = filepath + LOCK_FILE_SUFFIX
        self._lockrenamefile = self._lockfile + str(uuid.uuid4()) + RENAME_FILE_SUFFIX

    def acquire(self) -> bool:
        while True:
            try:
                os.symlink(self._lock_target_file, self._lockfile)
                return True
            except OSError as err:
                if err.errno == errno.EEXIST:
                    continue
                else:
                    raise err
            except BaseException:
                self.release()
                raise

    def release(self) -> None:
        try:
            os.rename(self._lockfile, self._lockrenamefile)
            os.unlink(self._lockrenamefile)
        except OSError:
            raise RuntimeError("Error: did not possess lock")
        except BaseException:
            os.unlink(self._lockrenamefile)
            raise


@experimental_class("3.1.0")
class JournalFileOpenLock(JournalFileBaseLock):
    """Lock class for synchronizing processes.

    On acquiring the lock, open system call is called with the O_EXCL option to create an exclusive
    file. The file is deleted when the lock is released. This class is only supported when using
    NFSv3 or later on kernel 2.6 or later. In prior NFS environments, use
    `~optuna.storages.JournalFileSymlinkLock`.

    Args:
        filepath:
            The path of the file whose race condition must be protected.
    """

    def __init__(self, filepath: str) -> None:
        self._lockfile = filepath + LOCK_FILE_SUFFIX
        self._lockrenamefile = self._lockfile + str(uuid.uuid4()) + RENAME_FILE_SUFFIX

    def acquire(self) -> bool:
        while True:
            try:
                open_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                os.close(os.open(self._lockfile, open_flags))
                return True
            except OSError as err:
                if err.errno == errno.EEXIST:
                    continue
                else:
                    raise err
            except BaseException:
                self.release()
                raise

    def release(self) -> None:
        try:
            os.rename(self._lockfile, self._lockrenamefile)
            os.unlink(self._lockrenamefile)
        except OSError:
            raise RuntimeError("Error: did not possess lock")
        except BaseException:
            os.unlink(self._lockrenamefile)
            raise


@contextmanager
def get_lock_file(lock_obj: JournalFileBaseLock) -> Iterator[None]:
    lock_obj.acquire()
    try:
        yield
    finally:
        lock_obj.release()


@experimental_class("3.1.0")
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
        open(self._file_path, "a").close()  # Create a file if it does not exist
        self._log_number_offset: Dict[int, int] = {0: 0}

    def read_logs(self, log_number_from: int) -> List[Dict[str, Any]]:
        with get_lock_file(self._lock):
            logs = []
            with open(self._file_path, "r") as f:
                log_number_start = 0
                if log_number_from in self._log_number_offset:
                    f.seek(self._log_number_offset[log_number_from])
                    log_number_start = log_number_from

                for log_number, line in enumerate(f, start=log_number_start):
                    if log_number + 1 not in self._log_number_offset:
                        byte_len = len(line.encode("utf-8"))
                        self._log_number_offset[log_number + 1] = (
                            self._log_number_offset[log_number] + byte_len
                        )
                    if log_number < log_number_from:
                        continue
                    logs.append(json.loads(line))

            return logs

    def append_logs(self, logs: List[Dict[str, Any]]) -> None:
        with get_lock_file(self._lock):
            what_to_write = ""
            for log in logs:
                what_to_write += json.dumps(log) + "\n"

            with open(self._file_path, "a", encoding="utf-8") as f:
                f.write(what_to_write)
                os.fsync(f.fileno())
