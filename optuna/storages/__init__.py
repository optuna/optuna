from optuna.storages.base import BaseStorage  # NOQA
from optuna.storages.in_memory import InMemoryStorage  # NOQA
from optuna.storages.redis import RedisStorage  # NOQA
from optuna.storages.rdb.storage import RDBStorage  # NOQA

from typing import Union  # NOQA


def get_storage(storage):
    # type: (Union[None, str, BaseStorage]) -> BaseStorage

    if storage is None:
        return InMemoryStorage()
    if isinstance(storage, str):
        if storage.startswith("redis"):
            return RedisStorage(storage)
        else:
            return RDBStorage(storage)
    else:
        return storage
