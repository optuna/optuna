import copy
from typing import Union

import optuna
from optuna._callbacks import RetryFailedTrialCallback  # NOQA
from optuna._experimental import experimental
from optuna.storages._base import BaseStorage
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._in_memory import InMemoryStorage
from optuna.storages._rdb.storage import RDBStorage
from optuna.storages._redis import RedisStorage


__all__ = [
    "BaseStorage",
    "InMemoryStorage",
    "RDBStorage",
    "RedisStorage",
    "_CachedStorage",
    "fail_stale_trials",
]


def get_storage(storage: Union[None, str, BaseStorage]) -> BaseStorage:
    """Only for internal usage. It might be deprecated in the future."""

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


@experimental("2.9.0")
def fail_stale_trials(study: "optuna.Study") -> None:
    """Fail stale trials and run their failure callbacks.

    The running trials whose heartbeat has not been updated for a long time will be failed,
    that is, those states will be changed to :obj:`~optuna.trial.TrialState.FAIL`.

    .. seealso::

       See :class:`~optuna.storages.RDBStorage`.

    Args:
        study:
            Study holding the trials to check.
    """
    storage = study._storage

    if not storage.is_heartbeat_enabled():
        return

    failed_trial_ids = storage.fail_stale_trials(study._study_id)
    failed_trial_callback = storage.get_failed_trial_callback()
    if failed_trial_callback is not None:
        for trial_id in failed_trial_ids:
            failed_trial = copy.deepcopy(storage.get_trial(trial_id))
            failed_trial_callback(study, failed_trial)
