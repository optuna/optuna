import json
import os
from typing import Any
from typing import Dict
from typing import List


class FileStorage:
    def __init__(self, file_name: str):
        self._filename: str = os.path.join(".", file_name)
        open(self._filename, "a").close()  # Create a file if it does not exist

    # TODO(wattlebirdaz): Use seek instead of readlines to achieve better performance.
    def get_unread_logs(self, log_number_read: int = 0) -> List[Dict[str, Any]]:
        # log_number starts from 1
        # The default log_number_read == 0 means no logs have been read by the caller
        with open(self._filename, "r") as f:
            lines = f.readlines()

        if len(lines) < log_number_read:
            raise RuntimeError("log_number too big")
        elif len(lines) == log_number_read:
            return []
        else:
            return [json.loads(line) for line in lines[log_number_read:]]

    def append_log(self, log: Dict[str, Any]) -> None:
        with open(self._filename, "a") as f:
            f.write(json.dumps(log) + "\n")
            os.fsync(f.fileno())

    def append_logs(self, logs: List[Dict[str, Any]]) -> None:
        what_to_write = ""
        for log in logs:
            what_to_write += json.dumps(log) + "\n"
        with open(self._filename, "a") as f:
            f.write(what_to_write)
            os.fsync(f.fileno())
