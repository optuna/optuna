import abc
import datetime
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

from optuna._deprecated import deprecated_func
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType


class BaseTrial(object, metaclass=abc.ABCMeta):
    """Base class for trials.

    Note that this class is not supposed to be directly accessed by library users.
    """

    @abc.abstractmethod
    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False,
    ) -> float:

        raise NotImplementedError

    @deprecated_func("3.0.0", "6.0.0")
    @abc.abstractmethod
    def suggest_uniform(self, name: str, low: float, high: float) -> float:

        raise NotImplementedError

    @deprecated_func("3.0.0", "6.0.0")
    @abc.abstractmethod
    def suggest_loguniform(self, name: str, low: float, high: float) -> float:

        raise NotImplementedError

    @deprecated_func("3.0.0", "6.0.0")
    @abc.abstractmethod
    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:

        raise NotImplementedError

    @abc.abstractmethod
    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:

        raise NotImplementedError

    @abc.abstractmethod
    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:

        raise NotImplementedError

    @abc.abstractmethod
    def report(self, value: float, step: int) -> None:

        raise NotImplementedError

    @abc.abstractmethod
    def should_prune(self) -> bool:

        raise NotImplementedError

    @abc.abstractmethod
    def set_user_attr(self, key: str, value: Any) -> None:

        raise NotImplementedError

    @abc.abstractmethod
    def set_system_attr(self, key: str, value: Any) -> None:

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def params(self) -> Dict[str, Any]:

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def distributions(self) -> Dict[str, BaseDistribution]:

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def user_attrs(self) -> Dict[str, Any]:

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def system_attrs(self) -> Dict[str, Any]:

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def datetime_start(self) -> Optional[datetime.datetime]:

        raise NotImplementedError

    @property
    def number(self) -> int:

        raise NotImplementedError
