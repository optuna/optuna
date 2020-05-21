from typing import Union  # NOQA

from optuna.storages.base import BaseStorage  # NOQA
from optuna.storages.cached_storage import _CachedStorage
from optuna.storages.in_memory import InMemoryStorage
from optuna.storages.rdb.storage import RDBStorage
from optuna.storages.redis import RedisStorage


def get_storage(storage):
    # type: (Union[None, str, BaseStorage]) -> BaseStorage

    if storage is None:
        return InMemoryStorage()
    if isinstance(storage, str):
        if storage.startswith("redis"):
            return RedisStorage(storage)
        else:
            return _CachedStorage(RDBStorage(storage))
    elif isinstance(storage, RDBStorage):
        return _CachedStorage(storage)
    else:
        return storage
