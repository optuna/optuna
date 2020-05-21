import tempfile

import fakeredis

import optuna
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from types import TracebackType  # NOQA
    from typing import Any  # NOQA
    from typing import IO  # NOQA
    from typing import Optional  # NOQA
    from typing import Type  # NOQA

SQLITE3_TIMEOUT = 300


class StorageSupplier(object):
    def __init__(self, storage_specifier):
        # type: (str) -> None

        self.storage_specifier = storage_specifier
        self.tempfile = None  # type: Optional[IO[Any]]

    def __enter__(self):
        # type: () -> optuna.storages.BaseStorage

        if self.storage_specifier == "inmemory":
            return optuna.storages.InMemoryStorage()
        elif self.storage_specifier == "sqlite":
            self.tempfile = tempfile.NamedTemporaryFile()
            url = "sqlite:///{}".format(self.tempfile.name)
            return optuna.storages.RDBStorage(
                url, engine_kwargs={"connect_args": {"timeout": SQLITE3_TIMEOUT}},
            )
        elif self.storage_specifier == "cache":
            self.tempfile = tempfile.NamedTemporaryFile()
            url = "sqlite:///{}".format(self.tempfile.name)
            return optuna.storages.cached_storage._CachedStorage(
                optuna.storages.RDBStorage(
                    url, engine_kwargs={"connect_args": {"timeout": SQLITE3_TIMEOUT}},
                )
            )
        elif self.storage_specifier == "redis":
            storage = optuna.storages.RedisStorage("redis://localhost")
            storage._redis = fakeredis.FakeStrictRedis()
            return storage
        else:
            assert False

    def __exit__(self, exc_type, exc_val, exc_tb):
        # type: (Type[BaseException], BaseException, TracebackType) -> None

        if self.tempfile:
            self.tempfile.close()
