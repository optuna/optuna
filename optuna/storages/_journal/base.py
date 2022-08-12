import abc
from typing import Any
from typing import Dict
from typing import List


class BaseJournalLogStorage(abc.ABC):
    """Base class for Journal storages.

    Storage classes implementing this base class must guarantee process safety. Thread-safety
    need not be guaranteed. Use `~optuna.storages.JournalFileLinkLock` or
    `~optuna.storages.JournalFileOpenLock` to create a critical section if your storage does not
    natively support multi-process query executions.

    """

    @abc.abstractmethod
    def read_logs(self, log_number_from: int) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def append_logs(self, logs: List[Dict[str, Any]]) -> None:
        raise NotImplementedError
