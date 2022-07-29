import json
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from optuna.storages._journal.base import BaseLogStorage
from optuna.storages._journal.file_lock import BaseFileLock
from optuna.storages._journal.file_lock import OpenLock


class FileStorage(BaseLogStorage):
    def __init__(self, file_path: str, lock_obj: Optional[BaseFileLock] = None) -> None:
        self._file_path: str = file_path
        open(self._file_path, "a").close()  # Create a file if it does not exist

        base_dir = os.path.dirname(file_path)
        lock_filename = os.path.basename(file_path) + ".lock"
        self._lock = lock_obj or OpenLock(base_dir, lock_filename)

    # TODO(wattlebirdaz): Use seek instead of readlines to achieve better performance.
    def get_unread_logs(self, log_number_read: int = 0) -> List[Dict[str, Any]]:
        # log_number starts from 1.
        # The default log_number_read == 0 means no logs have been read by the caller.
        self._lock.acquire()
        with open(self._file_path, "r") as f:
            lines = f.readlines()

        if len(lines) < log_number_read:
            self._lock.release()
            raise RuntimeError("log_number too big")
        elif len(lines) == log_number_read:
            self._lock.release()
            return []
        else:
            self._lock.release()
            return [json.loads(line) for line in lines[log_number_read:]]

    def append_logs(self, logs: List[Dict[str, Any]]) -> None:
        what_to_write = ""
        for log in logs:
            what_to_write += json.dumps(log) + "\n"

        self._lock.acquire()

        with open(self._file_path, "a") as f:
            f.write(what_to_write)
            os.fsync(f.fileno())

        self._lock.release()
