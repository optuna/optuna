import abc
import datetime
from typing import Optional

from optuna import distributions
from optuna import logging
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Sequence  # NOQA
    from typing import Union  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.distributions import CategoricalChoiceType  # NOQA
    from optuna.study import Study  # NOQA

    FloatingPointDistributionType = Union[
        distributions.UniformDistribution, distributions.LogUniformDistribution
    ]

_logger = logging.get_logger(__name__)


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
        log: bool = False
    ) -> float:

        raise NotImplementedError

    @abc.abstractmethod
    def suggest_uniform(self, name, low, high):
        # type: (str, float, float) -> float

        raise NotImplementedError

    @abc.abstractmethod
    def suggest_loguniform(self, name, low, high):
        # type: (str, float, float) -> float

        raise NotImplementedError

    @abc.abstractmethod
    def suggest_discrete_uniform(self, name, low, high, q):
        # type: (str, float, float, float) -> float

        raise NotImplementedError

    @abc.abstractmethod
    def suggest_int(self, name, low, high, step=1, log=False):
        # type: (str, int, int, int, bool) -> int

        raise NotImplementedError

    @abc.abstractmethod
    def suggest_categorical(self, name, choices):
        # type: (str, Sequence[CategoricalChoiceType]) -> CategoricalChoiceType

        raise NotImplementedError

    @abc.abstractmethod
    def report(self, value, step):
        # type: (float, int) -> None

        raise NotImplementedError

    @abc.abstractmethod
    def should_prune(self, step=None):
        # type: (Optional[int]) -> bool

        raise NotImplementedError

    @abc.abstractmethod
    def set_user_attr(self, key, value):
        # type: (str, Any) -> None

        raise NotImplementedError

    @abc.abstractmethod
    def set_system_attr(self, key, value):
        # type: (str, Any) -> None

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def params(self):
        # type: () -> Dict[str, Any]

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def distributions(self):
        # type: () -> Dict[str, BaseDistribution]

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def user_attrs(self):
        # type: () -> Dict[str, Any]

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def system_attrs(self):
        # type: () -> Dict[str, Any]

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def datetime_start(self):
        # type: () -> Optional[datetime.datetime]

        raise NotImplementedError

    @property
    def number(self) -> int:

        raise NotImplementedError
