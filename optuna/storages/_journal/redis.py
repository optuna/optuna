import json
from typing import Any
from typing import Dict
from typing import List

from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna.storages._journal.base import BaseJournalLogStorage


with try_import() as _imports:
    import redis


@experimental_class("3.1.0")
class JournalRedisStorage(BaseJournalLogStorage):
    """Redis storage class for Journal log backend.

    Args:
        url:
            URL of the redis storage, password and db are optional. (ie: redis://localhost:6379)

    """

    def __init__(self, url: str) -> None:

        _imports.check()

        self._url = url
        self._redis = redis.Redis.from_url(url)

    def read_logs(self, log_number_from: int) -> List[Dict[str, Any]]:

        if not self._redis.exists("log_number"):
            self._redis.set("log_number", -1)
        max_log_number = int(self._redis.get("log_number"))

        logs = []
        last_decode_error = None
        for log_number in range(log_number_from, max_log_number + 1):
            if last_decode_error is not None:
                raise last_decode_error
            log = self._redis.get(self._key_log_id(log_number)).decode("utf-8")
            try:
                logs.append(json.loads(log))
            except json.JSONDecodeError as err:
                last_decode_error = err
        return logs

    def append_logs(self, logs: List[Dict[str, Any]]) -> None:

        if not self._redis.exists("log_number"):
            self._redis.set("log_number", -1)

        for log in logs:
            log_number = self._redis.incr("log_number", 1)
            self._redis.set(self._key_log_id(log_number), json.dumps(log))

    @staticmethod
    def _key_log_id(log_number) -> str:
        return f"log:{log_number}"
