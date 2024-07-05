from __future__ import annotations

from contextlib import contextmanager
import errno
import os
import time
from typing import Iterator
import uuid

from optuna.storages.journal._base import BaseJournalFileLock


LOCK_FILE_SUFFIX = ".lock"
RENAME_FILE_SUFFIX = ".rename"


class JournalFileSymlinkLock(BaseJournalFileLock):
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

    def acquire(self) -> bool:
        """Acquire a lock in a blocking way by creating a symbolic link of a file.

        Returns:
            :obj:`True` if it succeeded in creating a symbolic link of ``self._lock_target_file``.

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

        lock_rename_file = self._lock_file + str(uuid.uuid4()) + RENAME_FILE_SUFFIX
        try:
            os.rename(self._lock_file, lock_rename_file)
            os.unlink(lock_rename_file)
        except OSError:
            raise RuntimeError("Error: did not possess lock")
        except BaseException:
            os.unlink(lock_rename_file)
            raise


class JournalFileOpenLock(BaseJournalFileLock):
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

    def acquire(self) -> bool:
        """Acquire a lock in a blocking way by creating a lock file.

        Returns:
            :obj:`True` if it succeeded in creating a ``self._lock_file``.

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

        lock_rename_file = self._lock_file + str(uuid.uuid4()) + RENAME_FILE_SUFFIX
        try:
            os.rename(self._lock_file, lock_rename_file)
            os.unlink(lock_rename_file)
        except OSError:
            raise RuntimeError("Error: did not possess lock")
        except BaseException:
            os.unlink(lock_rename_file)
            raise


@contextmanager
def get_lock_file(lock_obj: BaseJournalFileLock) -> Iterator[None]:
    lock_obj.acquire()
    try:
        yield
    finally:
        lock_obj.release()
