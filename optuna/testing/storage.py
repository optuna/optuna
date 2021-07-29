import tempfile
from types import TracebackType
from typing import Any
from typing import IO
from typing import Optional
from typing import Type

import fakeredis

import optuna


STORAGE_MODES = [
    "inmemory",
    "sqlite",
    "cache",
    "redis",
]

STORAGE_MODES_HEARTBEAT = [
    "sqlite",
    "cache",
    "redis",
]

SQLITE3_TIMEOUT = 300


class StorageSupplier(object):
    def __init__(self, storage_specifier: str, **kwargs: Any) -> None:

        self.storage_specifier = storage_specifier
        self.tempfile: Optional[IO[Any]] = None
        self.extra_args = kwargs

    def __enter__(self) -> optuna.storages.BaseStorage:

        if self.storage_specifier == "inmemory":
            if len(self.extra_args) > 0:
                raise ValueError("InMemoryStorage does not accept any arguments!")
            return optuna.storages.InMemoryStorage()
        elif self.storage_specifier == "sqlite":
            self.tempfile = tempfile.NamedTemporaryFile()
            url = "sqlite:///{}".format(self.tempfile.name)
            return optuna.storages.RDBStorage(
                url,
                engine_kwargs={"connect_args": {"timeout": SQLITE3_TIMEOUT}},
                **self.extra_args,
            )
        elif self.storage_specifier == "cache":
            self.tempfile = tempfile.NamedTemporaryFile()
            url = "sqlite:///{}".format(self.tempfile.name)
            return optuna.storages._CachedStorage(
                optuna.storages.RDBStorage(
                    url,
                    engine_kwargs={"connect_args": {"timeout": SQLITE3_TIMEOUT}},
                    **self.extra_args,
                )
            )
        elif self.storage_specifier == "redis":
            storage = optuna.storages.RedisStorage("redis://localhost", **self.extra_args)
            storage._redis = fakeredis.FakeStrictRedis()
            return storage
        else:
            assert False

    def __exit__(
        self, exc_type: Type[BaseException], exc_val: BaseException, exc_tb: TracebackType
    ) -> None:

        if self.tempfile:
            self.tempfile.close()
