import json
import os
from typing import Any
from typing import Dict
from typing import List
from file_lock import BaseFileLock
from file_lock import OpenLock


def get_file_lock(file_name) -> BaseFileLock:
    return OpenLock("./openlock", file_name)


class FileStorage:
    def __init__(self, file_name: str):
        self._filename: str = os.path.join(".", file_name)
        open(self._filename, "a").close()  # Create a file if it does not exist
        self._lock_file_name = file_name + "--lockfile"

    # TODO(wattlebirdaz): Use seek instead of readlines to achieve better performance.
    def get_unread_logs(self, log_number_read: int = 0) -> List[Dict[str, Any]]:
        # log_number starts from 1
        # The default log_number_read == 0 means no logs have been read by the caller
        lock = get_file_lock(self._lock_file_name)

        lock.acquire()
        with open(self._filename, "r") as f:
            lines = f.readlines()

        if len(lines) < log_number_read:
            lock.release()
            raise RuntimeError("log_number too big")
        elif len(lines) == log_number_read:
            lock.release()
            return []
        else:
            lock.release()
            return [json.loads(line) for line in lines[log_number_read:]]

    def append_logs(self, logs: List[Dict[str, Any]]) -> None:
        what_to_write = ""
        for log in logs:
            what_to_write += json.dumps(log) + "\n"

        lock = get_file_lock(self._lock_file_name)
        lock.acquire()

        with open(self._filename, "a") as f:
            f.write(what_to_write)
            os.fsync(f.fileno())

        lock.release()
