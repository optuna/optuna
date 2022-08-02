import abc
import errno
import json
import os
from types import TracebackType
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

from optuna.storages._journal.base import BaseJournalLogStorage


LOCK_FILE_SUFFIX = ".lock"


class BaseFileLock(abc.ABC):
    @abc.abstractmethod
    def acquire(self, blocking: bool = True) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def release(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def __enter__(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def __exit__(
        self,
        type: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        raise NotImplementedError


class LinkLock(BaseFileLock):
    def __init__(self, filepath: str) -> None:
        self._lock_target_file = filepath
        self._lockfile = filepath + LOCK_FILE_SUFFIX

        try:
            os.makedirs(os.path.dirname(self._lockfile))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise RuntimeError("Error: mkdir")

        open(self._lock_target_file, "a").close()  # Create file if it does not exist

    def acquire(self, blocking: bool = True) -> bool:
        if blocking:
            while True:
                try:
                    os.link(self._lock_target_file, self._lockfile)
                    return True
                except OSError as err:
                    if err.errno == errno.EEXIST or err.errno == errno.ENOENT:
                        continue
                    else:
                        raise err
        else:
            try:
                os.link(self._lock_target_file, self._lockfile)
                return True
            except OSError as err:
                if err.errno == errno.EEXIST or err.errno == errno.ENOENT:
                    return False
                else:
                    raise err

    def release(self) -> None:
        try:
            os.unlink(self._lockfile)
        except OSError:
            raise RuntimeError("Error: did not possess lock")

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(
        self,
        type: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        return self.release()


class OpenLock(BaseFileLock):
    def __init__(self, filepath: str) -> None:
        self._lockfile = filepath + LOCK_FILE_SUFFIX

        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise RuntimeError("Error: mkdir")

    def acquire(self, blocking: bool = True) -> bool:
        if blocking:
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
        else:
            try:
                open_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
                os.close(os.open(self._lockfile, open_flags))
                return True
            except OSError as err:
                if err.errno == errno.EEXIST:
                    return False
                else:
                    raise err

    def release(self) -> None:
        try:
            os.unlink(self._lockfile)
        except OSError:
            raise RuntimeError("Error: did not possess lock")

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(
        self,
        type: Optional[Type[BaseException]],
        value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        return self.release()


class JournalFileStorage(BaseJournalLogStorage):
    def __init__(self, file_path: str, lock_obj: Optional[BaseFileLock] = None) -> None:
        self._file_path: str = file_path
        self._lock = lock_obj or OpenLock(self._file_path)
        open(self._file_path, "a").close()  # Create a file if it does not exist

    # TODO(wattlebirdaz): Use seek instead of readlines to achieve better performance.
    def get_unread_logs(self, log_number_read: int) -> List[Dict[str, Any]]:
        # log_number starts from 1.
        # The default log_number_read == 0 means no logs have been read by the caller.

        with self._lock:
            with open(self._file_path, "r") as f:
                lines = f.readlines()

            assert len(lines) >= log_number_read
            return [json.loads(line) for line in lines[log_number_read:]]

    def append_logs(self, logs: List[Dict[str, Any]]) -> None:
        what_to_write = ""
        for log in logs:
            what_to_write += json.dumps(log) + "\n"

        with self._lock:
            with open(self._file_path, "a") as f:
                f.write(what_to_write)
                os.fsync(f.fileno())
