import abc
import errno
import os


class BaseFileLock(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def acquire(self, blocking: bool) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def release(self) -> None:
        raise NotImplementedError


class LinkLock(BaseFileLock):
    def __init__(self, dir: str, lockfile: str) -> None:
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise RuntimeError("Error: mkdir")
        self._lockfile = os.path.join(dir, lockfile)
        # This file is common among all the threads
        self._linkfile = os.path.join(dir, lockfile + "-singlelinkfile")

        open(self._lockfile, "a").close()  # Create file if it does not exist

    def acquire(self, blocking: bool = True) -> bool:
        if blocking:
            while True:
                try:
                    os.link(self._lockfile, self._linkfile)
                    return True
                except OSError as err:
                    if err.errno == errno.EEXIST:
                        continue
                    else:
                        raise err
        else:
            try:
                os.link(self._lockfile, self._linkfile)
                return True
            except OSError as err:
                if err.errno == errno.EEXIST:
                    return False
                else:
                    raise err

    def release(self) -> None:
        try:
            os.unlink(self._linkfile)
        except OSError:
            raise RuntimeError("Error: did not possess lock")


class OpenLock(BaseFileLock):
    def __init__(self, dir: str, lockfile: str) -> None:
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise RuntimeError("Error: mkdir")
        self._lockfile = os.path.join(dir, lockfile)

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
