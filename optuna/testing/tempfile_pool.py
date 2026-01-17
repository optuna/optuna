# On Windows, temporary file shold delete "after" storage was deleted
# NamedTemporaryFilePool ensures tempfile delete after tests.

from __future__ import annotations

import atexit
import copy
import os
import tempfile
import threading
from types import TracebackType
from typing import Any
from typing import cast
from typing import ClassVar
from typing import IO


class NamedTemporaryFilePool:
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _path: ClassVar[list[str]] = []
    _registered: ClassVar[bool] = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._file: IO[bytes] | IO[str] | None = None

        with self.__class__._lock:
            if not self.__class__._registered:
                _ = atexit.register(self.__class__.cleanup)
                self.__class__._registered = True

    def __enter__(self) -> IO[bytes] | IO[str]:
        return self.tempfile()

    def tempfile(self) -> IO[bytes] | IO[str]:
        f = cast("IO[bytes] | IO[str]", tempfile.NamedTemporaryFile(delete=False, **self.kwargs))
        self._file = f
        with self.__class__._lock:
            self.__class__._path.append(f.name)
        return self._file

    def __exit__(
        self,
        exc_type: type[BaseException],
        exc_val: BaseException,
        exc_tb: TracebackType,
    ) -> None:
        if self._file is not None:
            self._file.close()

    @classmethod
    def cleanup(cls) -> None:
        with cls._lock:
            path = copy.deepcopy(cls._path)
            cls._path = []

        for p in path:
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
            except PermissionError:
                pass
