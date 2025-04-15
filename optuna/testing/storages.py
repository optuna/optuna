from __future__ import annotations

import socket
import threading
from types import TracebackType
from typing import Any
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


STORAGE_MODES_DIRECT: list[Any] = [
    "inmemory",
    "sqlite",
    "cached_sqlite",
    "journal",
    "journal_redis",
]

STORAGE_MODES_GRPC = ["grpc_" + mode for mode in STORAGE_MODES_DIRECT]

STORAGE_MODES = STORAGE_MODES_DIRECT + STORAGE_MODES_GRPC

STORAGE_MODES_HEARTBEAT = [
    "sqlite",
    "cached_sqlite",
]


STORAGE_MODES_PAIRS = [
    pair
    for pair in zip(
        STORAGE_MODES_DIRECT + STORAGE_MODES_GRPC, STORAGE_MODES_GRPC + STORAGE_MODES_DIRECT
    )
    if "inmemory" not in pair[0]
]
SQLITE3_TIMEOUT = 300


class StorageSupplier:
    def __init__(self, storage_specifier: str, **kwargs: Any) -> None:
        self.storage_specifier = storage_specifier
        self.extra_args = kwargs
        self.tempfile: IO[Any] | None = None
        self.server: grpc.Server | None = None
        self.thread: threading.Thread | None = None
        self.proxy: GrpcStorageProxy | None = None

    def __enter__(self) -> BaseStorage:
        storage: BaseStorage = self._create_direct_storage()
        if "cached_" in self.storage_specifier:
            assert isinstance(storage, optuna.storages.RDBStorage)
            storage = self._create_cached_storage(storage)
        elif "grpc_" in self.storage_specifier:
            storage = self._create_proxy(storage)
        return storage

    def _create_cached_storage(
        self, storage: optuna.storages.RDBStorage
    ) -> optuna.storages._CachedStorage:
        return optuna.storages._CachedStorage(storage)

    def _create_direct_storage(
        self,
    ) -> (
        optuna.storages.InMemoryStorage
        | optuna.storages.RDBStorage
        | optuna.storages.JournalStorage
    ):
        if "inmemory" in self.storage_specifier:
            if len(self.extra_args) > 0:
                raise ValueError("InMemoryStorage does not accept any arguments!")
            return optuna.storages.InMemoryStorage()
        elif "sqlite" in self.storage_specifier:
            self.tempfile = NamedTemporaryFilePool().tempfile()
            url = "sqlite:///{}".format(self.tempfile.name)
            return optuna.storages.RDBStorage(
                url,
                engine_kwargs={"connect_args": {"timeout": SQLITE3_TIMEOUT}},
                **self.extra_args,
            )
        elif "journal_redis" in self.storage_specifier:
            journal_redis_storage = optuna.storages.journal.JournalRedisBackend(
                "redis://localhost"
            )
            journal_redis_storage._redis = self.extra_args.get(
                "redis", fakeredis.FakeStrictRedis()  # type: ignore[no-untyped-call]
            )
            return optuna.storages.JournalStorage(journal_redis_storage)
        elif "journal" in self.storage_specifier:
            self.tempfile = self.extra_args.get("file", NamedTemporaryFilePool().tempfile())
            assert self.tempfile is not None
            file_storage = JournalFileBackend(self.tempfile.name)
            return optuna.storages.JournalStorage(file_storage)
        else:
            assert False, "Unsupported storage specifier: {}".format(self.storage_specifier)

    def _create_proxy(self, storage: BaseStorage) -> GrpcStorageProxy:
        port = _find_free_port()
        self.server = optuna.storages._grpc.server.make_server(storage, "localhost", port)
        self.thread = threading.Thread(target=self.server.start)
        self.thread.start()
        self.proxy = GrpcStorageProxy(host="localhost", port=port)
        self.proxy.wait_server_ready(timeout=60)
        return self.proxy

    def __exit__(
        self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType
    ) -> None:
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
