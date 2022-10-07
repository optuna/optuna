from typing import Union

from optuna._callbacks import RetryFailedTrialCallback
from optuna.storages._base import BaseStorage
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._heartbeat import fail_stale_trials
from optuna.storages._in_memory import InMemoryStorage
from optuna.storages._journal.base import BaseJournalLogStorage
from optuna.storages._journal.file import JournalFileOpenLock
from optuna.storages._journal.file import JournalFileStorage
from optuna.storages._journal.file import JournalFileSymlinkLock
from optuna.storages._journal.storage import JournalStorage
from optuna.storages._rdb.storage import RDBStorage
from optuna.storages._redis import RedisStorage
from optuna.storages._gcp_datastore import DatastoreStorage


__all__ = [
    "BaseStorage",
    "BaseJournalLogStorage",
    "InMemoryStorage",
    "RDBStorage",
    "RedisStorage",
    "DatastoreStorage",
    "JournalStorage",
    "JournalFileSymlinkLock",
    "JournalFileOpenLock",
    "JournalFileStorage",
    "RetryFailedTrialCallback",
    "_CachedStorage",
    "fail_stale_trials",
]


def get_storage(storage: Union[None, str, BaseStorage]) -> BaseStorage:
    """Only for internal usage. It might be deprecated in the future."""

    if storage is None:
        return InMemoryStorage()
    if isinstance(storage, str):
        if storage.startswith("redis"):
            return _CachedStorage(RedisStorage(storage))
        elif storage == "datastore":
            return DatastoreStorage()
        else:
            return _CachedStorage(RDBStorage(storage))
    elif isinstance(storage, (RDBStorage, RedisStorage)):
        return _CachedStorage(storage)
    else:
        return storage
