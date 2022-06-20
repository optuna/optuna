import tempfile
from types import TracebackType
from typing import Any
from typing import IO
from typing import Optional
from typing import Type
from typing import Union

import fakeredis

import optuna


STORAGE_MODES = [
    "inmemory",
    "sqlite",
    "cached_sqlite",
    "redis",
    "cached_redis",
]

STORAGE_MODES_HEARTBEAT = [
    "sqlite",
    "cached_sqlite",
    "redis",
    "cached_redis",
]

SQLITE3_TIMEOUT = 300


class StorageSupplier:
    def __init__(self, storage_specifier: str, **kwargs: Any) -> None:

        self.storage_specifier = storage_specifier
        self.tempfile: Optional[IO[Any]] = None
        self.extra_args = kwargs

    def __enter__(
        self,
    ) -> Union[
        optuna.storages.InMemoryStorage,
        optuna.storages._CachedStorage,
        optuna.storages.RDBStorage,
        optuna.storages.RedisStorage,
    ]:

        if self.storage_specifier == "inmemory":
            if len(self.extra_args) > 0:
                raise ValueError("InMemoryStorage does not accept any arguments!")
            return optuna.storages.InMemoryStorage()
        elif "sqlite" in self.storage_specifier:
            self.tempfile = tempfile.NamedTemporaryFile()
            url = "sqlite:///{}".format(self.tempfile.name)
            rdb_storage = optuna.storages.RDBStorage(
                url,
                engine_kwargs={"connect_args": {"timeout": SQLITE3_TIMEOUT}},
                **self.extra_args,
            )
            return (
                optuna.storages._CachedStorage(rdb_storage)
                if "cached" in self.storage_specifier
                else rdb_storage
            )
        elif "redis" in self.storage_specifier:
            redis_storage = optuna.storages.RedisStorage("redis://localhost", **self.extra_args)
            redis_storage._redis = fakeredis.FakeStrictRedis()
            return (
                optuna.storages._CachedStorage(redis_storage)
                if "cached" in self.storage_specifier
                else redis_storage
            )
        else:
            assert False

    def __exit__(
        self, exc_type: Type[BaseException], exc_val: BaseException, exc_tb: TracebackType
    ) -> None:

        if self.tempfile:
            self.tempfile.close()
