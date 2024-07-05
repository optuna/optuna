import warnings

from optuna._deprecated import _DEPRECATION_WARNING_TEMPLATE
from optuna.storages._journal.backend import JournalFileStorage
from optuna.storages._journal.base import JournalFileBaseLock
from optuna.storages._journal.file_lock import get_lock_file
from optuna.storages._journal.file_lock import JournalFileOpenLock
from optuna.storages._journal.file_lock import JournalFileSymlinkLock
from optuna.storages._journal.file_lock import LOCK_FILE_SUFFIX
from optuna.storages._journal.file_lock import RENAME_FILE_SUFFIX


msg = _DEPRECATION_WARNING_TEMPLATE.format(
    name="optuna.storages._journal.file.py", d_ver="4.0.0", r_ver="5.0.0"
)
warnings.warn(msg)


__all__ = [
    "JournalFileBaseLock",
    "JournalFileSymlinkLock",
    "JournalFileOpenLock",
    "get_lock_file",
    "JournalFileStorage",
    "LOCK_FILE_SUFFIX",
    "RENAME_FILE_SUFFIX",
]
