from __future__ import annotations

import socket
import threading
from types import TracebackType
from typing import Any
from typing import IO
from typing import TYPE_CHECKING

import fakeredis

import optuna
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
    "grpc",
]


STORAGE_MODES_HEARTBEAT = [
    "sqlite",
    "cached_sqlite",
]

SQLITE3_TIMEOUT = 300


class StorageSupplier:
    def __init__(self, storage_specifier: str, **kwargs: Any) -> None:
        self.storage_specifier = storage_specifier
        self.extra_args = kwargs
        self.tempfile: IO[Any] | None = None
        self.server: grpc.Server | None = None
        self.thread: threading.Thread | None = None
        self.storage: optuna.storages.BaseStorage | None = None

    def __enter__(
        self,
    ) -> optuna.storages.BaseStorage:
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
                "redis", fakeredis.FakeStrictRedis()  # type: ignore[no-untyped-call]
            )
            self.storage = optuna.storages.JournalStorage(journal_redis_storage)
        elif "journal" in self.storage_specifier:
            self.tempfile = self.extra_args.get("file", NamedTemporaryFilePool().tempfile())
            assert self.tempfile is not None
            file_storage = JournalFileBackend(self.tempfile.name)
            self.storage = optuna.storages.JournalStorage(file_storage)
        elif self.storage_specifier == "grpc":
            self.tempfile = NamedTemporaryFilePool().tempfile()
            url = "sqlite:///{}".format(self.tempfile.name)
            port = _find_free_port()

            self.server = optuna.storages._grpc.server.make_server(
                optuna.storages.RDBStorage(url), "localhost", port
            )
            self.thread = threading.Thread(target=self.server.start)
            self.thread.start()

            self.storage = GrpcStorageProxy(host="localhost", port=port, timeout=60)
        else:
            assert False
        assert self.storage is not None
        return self.storage

    def __exit__(
        self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType
    ) -> None:
        if self.tempfile:
            self.tempfile.close()

        if self.storage:
            del self.storage
            self.storage = None

        if self.server:
            assert self.thread is not None
            self.server.stop(5).wait()
            self.thread.join()
            self.server = None
            self.thread = None


def _find_free_port() -> int:
    for port in range(13000, 13100):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("localhost", port))
            return port
        except OSError:
            continue
    assert False, "must not reach here"
