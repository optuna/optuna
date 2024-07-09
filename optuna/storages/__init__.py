from typing import Union

from optuna._deprecated import deprecated_class
from optuna.storages._base import BaseStorage
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._callbacks import RetryFailedTrialCallback
from optuna.storages._heartbeat import fail_stale_trials
from optuna.storages._in_memory import InMemoryStorage
from optuna.storages._rdb.storage import RDBStorage
from optuna.storages.journal._backend import JournalFileStorage
from optuna.storages.journal._base import BaseJournalLogStorage
from optuna.storages.journal._file_lock import JournalFileOpenLock as _JournalFileOpenLock
from optuna.storages.journal._file_lock import JournalFileSymlinkLock as _JournalFileSymlinkLock
from optuna.storages.journal._redis import JournalRedisStorage
from optuna.storages.journal._storage import JournalStorage


@deprecated_class(
    deprecated_version="4.0.0",
    removed_version="6.0.0",
    name="The import path :class:`~optuna.storages.JournalFileOpenLock`",
    text="Use :class:`~optuna.storages.journal.JournalFileOpenLock` instead.",
)
class JournalFileOpenLock(_JournalFileOpenLock):
    """Lock class for synchronizing processes for NFSv3 or later.

    On acquiring the lock, open system call is called with the O_EXCL option to create an exclusive
    file. The file is deleted when the lock is released. This class is only supported when using
    NFSv3 or later on kernel 2.6 or later. In prior NFS environments, use
    :class:`~optuna.storages.journal.JournalFileSymlinkLock`.

    Args:
        filepath:
            The path of the file whose race condition must be protected.

    .. note::

        The path of :class:`~optuna.storages.JournalFileOpenLock` has been moved to
        :class:`~optuna.storages.journal.JournalFileOpenLock`.

    """

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath=filepath)


@deprecated_class(
    deprecated_version="4.0.0",
    removed_version="6.0.0",
    name="The import path :class:`~optuna.storages.JournalFileSymlinkLock`",
    text="Use :class:`~optuna.storages.journal.JournalFileSymlinkLock` instead.",
)
class JournalFileSymlinkLock(_JournalFileSymlinkLock):
    """Lock class for synchronizing processes for NFSv2 or later.

    On acquiring the lock, link system call is called to create an exclusive file. The file is
    deleted when the lock is released. In NFS environments prior to NFSv3, use this instead of
    :class:`~optuna.storages.journal.JournalFileOpenLock`.

    Args:
        filepath:
            The path of the file whose race condition must be protected.

    .. note::

        The path of :class:`~optuna.storages.JournalFileSymlinkLock` has been moved to
        :class:`~optuna.storages.journal.JournalFileSymlinkLock`.

    """

    def __init__(self, filepath: str) -> None:
        super().__init__(filepath=filepath)


__all__ = [
    "BaseStorage",
    "BaseJournalLogStorage",
    "InMemoryStorage",
    "RDBStorage",
    "JournalStorage",
    "JournalFileStorage",
    "JournalRedisStorage",
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
            raise ValueError(
                "RedisStorage is removed at Optuna v3.1.0. Please use JournalRedisBackend instead."
            )
        return _CachedStorage(RDBStorage(storage))
    elif isinstance(storage, RDBStorage):
        return _CachedStorage(storage)
    else:
        return storage
