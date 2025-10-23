from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from contextlib import contextmanager
import os
import socket
import sys
import threading
from types import TracebackType
from typing import Any
from typing import Generator
from typing import IO
from typing import TYPE_CHECKING

import fakeredis

import optuna
from optuna.storages import BaseStorage
from optuna.storages import GrpcStorageProxy
from optuna.storages.journal import JournalFileBackend
from optuna.testing.tempfile_pool import NamedTemporaryFilePool


if TYPE_CHECKING:
    import grpc
else:
    from optuna._imports import _LazyImport

    grpc = _LazyImport("grpc")


STORAGE_MODES: list[Any] = [
    "inmemory",
    "sqlite",
    "cached_sqlite",
    "journal",
    "journal_redis",
    "grpc_rdb",
    "grpc_journal_file",
]


STORAGE_MODES_HEARTBEAT = [
    "sqlite",
    "cached_sqlite",
]

SQLITE3_TIMEOUT = 300


@contextmanager
def _lock_to_search_for_free_port() -> Generator[None, None, None]:
    if sys.platform == "win32":
        lock_path = os.path.join(
            os.environ.get("PROGRAMDATA", "C:\\ProgramData"),
            "optuna",
            "optuna_find_free_port.lock",
        )
    else:
        lock_path = "/tmp/optuna_find_free_port.lock"

    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lockfile = open(lock_path, "w")
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(lockfile.fileno(), msvcrt.LK_LOCK, 1)
        yield
        msvcrt.locking(lockfile.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(lockfile, fcntl.LOCK_EX)
        yield
        fcntl.flock(lockfile, fcntl.LOCK_UN)

    lockfile.close()


class StorageSupplier(AbstractContextManager):
    def __init__(self, storage_specifier: str, **kwargs: Any) -> None:
        self.storage_specifier = storage_specifier
        self.extra_args = kwargs
        self.tempfile: IO[Any] | None = None
        self.server: grpc.Server | None = None
        self.thread: threading.Thread | None = None
        self.proxy: GrpcStorageProxy | None = None
        self.storage: BaseStorage | None = None
        self.backend_storage: BaseStorage | None = None

    def __enter__(
        self,
    ) -> (
        optuna.storages.InMemoryStorage
        | optuna.storages._CachedStorage
        | optuna.storages.RDBStorage
        | optuna.storages.JournalStorage
        | optuna.storages.GrpcStorageProxy
    ):
        if self.storage_specifier == "inmemory":
            if len(self.extra_args) > 0:
                raise ValueError("InMemoryStorage does not accept any arguments!")
            self.storage = optuna.storages.InMemoryStorage()
        elif "sqlite" in self.storage_specifier:
            self.tempfile = NamedTemporaryFilePool().tempfile()
            url = "sqlite:///{}".format(self.tempfile.name)
            rdb_storage = optuna.storages.RDBStorage(
                url,
                engine_kwargs={"connect_args": {"timeout": SQLITE3_TIMEOUT}},
                **self.extra_args,
            )
            self.storage = (
                optuna.storages._CachedStorage(rdb_storage)
                if "cached" in self.storage_specifier
                else rdb_storage
            )
        elif self.storage_specifier == "journal_redis":
            journal_redis_storage = optuna.storages.journal.JournalRedisBackend(
                "redis://localhost"
            )
            journal_redis_storage._redis = self.extra_args.get(
                "redis", fakeredis.FakeStrictRedis()
            )
            self.storage = optuna.storages.JournalStorage(journal_redis_storage)
        elif self.storage_specifier == "grpc_journal_file":
            self.tempfile = self.extra_args.get("file", NamedTemporaryFilePool().tempfile())
            assert self.tempfile is not None
            storage = optuna.storages.JournalStorage(
                optuna.storages.journal.JournalFileBackend(self.tempfile.name)
            )
            self.storage = self._create_proxy(
                storage, thread_pool=self.extra_args.get("thread_pool")
            )
        elif "journal" in self.storage_specifier:
            self.tempfile = self.extra_args.get("file", NamedTemporaryFilePool().tempfile())
            assert self.tempfile is not None
            file_storage = JournalFileBackend(self.tempfile.name)
            self.storage = optuna.storages.JournalStorage(file_storage)
        elif self.storage_specifier == "grpc_rdb":
            self.tempfile = NamedTemporaryFilePool().tempfile()
            url = "sqlite:///{}".format(self.tempfile.name)
            self.backend_storage = optuna.storages.RDBStorage(url)
            self.storage = self._create_proxy(self.backend_storage)
        elif self.storage_specifier == "grpc_proxy":
            assert "base_storage" in self.extra_args
            self.storage = self._create_proxy(self.extra_args["base_storage"])
        else:
            assert False

        return self.storage

    def _create_proxy(
        self, storage: BaseStorage, thread_pool: ThreadPoolExecutor | None = None
    ) -> GrpcStorageProxy:
        with _lock_to_search_for_free_port():
            port = _find_free_port()
            self.server = optuna.storages._grpc.server.make_server(
                storage, "localhost", port, thread_pool=thread_pool
            )
            self.thread = threading.Thread(target=self.server.start)
            self.thread.start()
            self.proxy = GrpcStorageProxy(host="localhost", port=port)
            self.proxy.wait_server_ready(timeout=60)
            return self.proxy

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        # Unit tests create many short-lived Engine objects, so the connections created by the
        # engine should be explicitly closed.
        if isinstance(self.storage, optuna.storages.RDBStorage):
            self.storage.engine.dispose()
        elif isinstance(self.storage, optuna.storages._CachedStorage):
            self.storage._backend.engine.dispose()
        elif self.storage_specifier == "grpc_rdb":
            assert isinstance(self.backend_storage, optuna.storages.RDBStorage)
            self.backend_storage.engine.dispose()

        if self.tempfile:
            self.tempfile.close()

        if self.proxy:
            self.proxy.close()
            self.proxy = None

        if self.server:
            assert self.thread is not None
            self.server.stop(5).wait()
            self.thread.join()
            self.server = None
            self.thread = None


def _find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for port in range(13000, 13100):
        try:
            sock.bind(("localhost", port))
            return port
        except OSError:
            continue
    assert False, "must not reach here"
