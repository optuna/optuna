import abc
from typing import Any
from typing import Dict
from typing import List


class BaseLogStorage(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_unread_logs(self, log_number_read: int = 0) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def append_logs(self, logs: List[Dict[str, Any]]) -> None:
        raise NotImplementedError
