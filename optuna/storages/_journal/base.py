import abc
from typing import Any
from typing import Dict
from typing import List


class BaseJournalLogStorage(abc.ABC):
    @abc.abstractmethod
    def read_logs(self, log_number_from: int) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def append_logs(self, logs: List[Dict[str, Any]]) -> None:
        raise NotImplementedError
