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

from optuna._experimental import experimental_class
from optuna.storages._journal.base import BaseJournalLogStorage


LOCK_FILE_SUFFIX = ".lock"


class BaseFileLock(abc.ABC):
    @abc.abstractmethod
    def acquire(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def release(self) -> None:
        raise NotImplementedError


class LinkLock(BaseFileLock):
    def __init__(self, filepath: str) -> None:
        self._lock_target_file = filepath
        self._lockfile = filepath + LOCK_FILE_SUFFIX

    def acquire(self) -> bool:
        while True:
            try:
                os.link(self._lock_target_file, self._lockfile)
                return True
            except OSError as err:
                if err.errno == errno.EEXIST or err.errno == errno.ENOENT:
                    continue
                else:
                    raise err
            except BaseException:
                os.unlink(self._lockfile)
                raise

    def release(self) -> None:
        try:
            os.unlink(self._lockfile)
        except OSError as e:
            raise RuntimeError("Error: did not possess lock")


class OpenLock(BaseFileLock):
    def __init__(self, filepath: str) -> None:
        self._lockfile = filepath + LOCK_FILE_SUFFIX

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
                os.unlink(self._lockfile)
                raise

    def release(self) -> None:
        try:
            os.unlink(self._lockfile)
        except OSError:
            raise RuntimeError("Error: did not possess lock")


@contextmanager
def get_lock_file(lock_obj: BaseFileLock) -> Iterator[None]:
    lock_obj.acquire()
    try:
        yield
    finally:
        lock_obj.release()


@experimental_class("3.1.0")
class JournalFileStorage(BaseJournalLogStorage):
    def __init__(self, file_path: str, lock_obj: Optional[BaseFileLock] = None) -> None:
        self._file_path: str = file_path
        self._lock = lock_obj or OpenLock(self._file_path)
        open(self._file_path, "a").close()  # Create a file if it does not exist

    def get_unread_logs(self, log_number_read: int) -> List[Dict[str, Any]]:
        # log_number starts from 1.
        # The default log_number_read == 0 means no logs have been read by the caller.

        with get_lock_file(self._lock):
            logs = []
            with open(self._file_path, "r") as f:
                for lineno, line in enumerate(f):
                    if lineno < log_number_read:
                        continue
                    logs.append(json.loads(line))
            return logs

    def append_logs(self, logs: List[Dict[str, Any]]) -> None:
        what_to_write = ""
        for log in logs:
            what_to_write += json.dumps(log) + "\n"

        with get_lock_file(self._lock):
            with open(self._file_path, "a") as f:
                f.write(what_to_write)
                os.fsync(f.fileno())
