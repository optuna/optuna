from typing import Union
import warnings

from optuna._callbacks import RetryFailedTrialCallback  # NOQA
from optuna.exceptions import ExperimentalWarning
from optuna.storages._base import BaseStorage
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._in_memory import InMemoryStorage
from optuna.storages._in_memory import MultiprocessInMemoryStorage
from optuna.storages._rdb.storage import RDBStorage
from optuna.storages._redis import RedisStorage


__all__ = [
    "BaseStorage",
    "InMemoryStorage",
    "MultiprocessInMemoryStorage",
    "RDBStorage",
    "RedisStorage",
    "_CachedStorage",
]


def get_storage(storage: Union[None, str, BaseStorage]) -> BaseStorage:
    """Only for internal usage. It might be deprecated in the future."""

    if storage is None:
        return InMemoryStorage()
    if isinstance(storage, str):
        if storage == "multiprocess":
            warnings.warn(
                "``multiprocess`` storage is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )
            return MultiprocessInMemoryStorage()
        if storage.startswith("redis"):
            return RedisStorage(storage)
        else:
            return _CachedStorage(RDBStorage(storage))
    elif isinstance(storage, RDBStorage):
        return _CachedStorage(storage)
    else:
        return storage
