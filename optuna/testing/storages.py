from __future__ import annotations

import sys
import tempfile
from types import TracebackType
from typing import Any
from typing import IO
from typing import Optional
from typing import Type
from typing import Union

import fakeredis
import pytest

import optuna
from optuna.storages import JournalFileStorage


try:
    import distributed

    _has_distributed = True
except ImportError:
    _has_distributed = False

STORAGE_MODES: list[Any] = [
    "inmemory",
    "sqlite",
    "cached_sqlite",
    "journal",
    "journal_redis",
    pytest.param(
        "dask",
        marks=[
            pytest.mark.integration,
            pytest.mark.skipif(
                sys.version_info[:2] >= (3, 11),
                reason="distributed doesn't yet support Python 3.11",
            ),
        ],
    ),
]


STORAGE_MODES_HEARTBEAT = [
    "sqlite",
    "cached_sqlite",
]

SQLITE3_TIMEOUT = 300


class StorageSupplier:
    def __init__(self, storage_specifier: str, **kwargs: Any) -> None:
        self.storage_specifier = storage_specifier
        self.tempfile: Optional[IO[Any]] = None
        self.extra_args = kwargs
        self.dask_client: Optional["distributed.Client"] = None

    def __enter__(
        self,
    ) -> Union[
        optuna.storages.InMemoryStorage,
        optuna.storages._CachedStorage,
        optuna.storages.RDBStorage,
        optuna.storages.JournalStorage,
        "optuna.integration.DaskStorage",
    ]:
        self.storage: Union[
            optuna.storages.InMemoryStorage,
            optuna.storages._CachedStorage,
            optuna.storages.RDBStorage,
            optuna.storages.JournalStorage,
            "optuna.integration.DaskStorage",
        ]
        if self.storage_specifier == "inmemory":
            if len(self.extra_args) > 0:
                raise ValueError("InMemoryStorage does not accept any arguments!")
            self.storage = optuna.storages.InMemoryStorage()
            return self.storage
        elif "sqlite" in self.storage_specifier:
            self.tempfile = tempfile.NamedTemporaryFile()
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
            return self.storage
        elif self.storage_specifier == "journal_redis":
            journal_redis_storage = optuna.storages.JournalRedisStorage("redis://localhost")
            journal_redis_storage._redis = self.extra_args.get(
                "redis", fakeredis.FakeStrictRedis()
            )
            self.storage = optuna.storages.JournalStorage(journal_redis_storage)
            return self.storage
        elif "journal" in self.storage_specifier:
            self.tempfile = tempfile.NamedTemporaryFile(delete=False)
            file_storage = JournalFileStorage(self.tempfile.name)
            self.storage = optuna.storages.JournalStorage(file_storage)
            return self.storage
        elif self.storage_specifier == "dask":
            self.dask_client = distributed.Client()  # type: ignore[no-untyped-call]

            self.storage = optuna.integration.DaskStorage(
                client=self.dask_client, **self.extra_args
            )
            return self.storage
        else:
            assert False

    def __exit__(
        self, exc_type: Type[BaseException], exc_val: BaseException, exc_tb: TracebackType
    ) -> None:
        if isinstance(self.storage, optuna.storages.RDBStorage):
            self.storage.engine.dispose()  # Detach process from tempfile before tempfile deletion.

        if self.tempfile:
            self.tempfile.close()

        if self.dask_client:
            self.dask_client.shutdown()  # type: ignore[no-untyped-call]
            self.dask_client.close()  # type: ignore[no-untyped-call]
