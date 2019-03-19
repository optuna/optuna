from optuna.storages.base import BaseStorage  # NOQA
from optuna.storages.in_memory import InMemoryStorage  # NOQA
from optuna.storages.rdb.storage import RDBStorage  # NOQA

from typing import Union  # NOQA


def get_storage(storage, enable_storage_cache=True):
    # type: (Union[None, str, BaseStorage], bool) -> BaseStorage

    if storage is None:
        return InMemoryStorage()
    if isinstance(storage, str):
        return RDBStorage(storage, enable_storage_cache=enable_storage_cache)
    else:
        return storage
