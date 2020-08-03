import abc
from typing import Any
from typing import Dict
from typing import List

from optuna._experimental import experimental
from optuna.study import Study
from optuna.trial import FrozenTrial


@experimental("2.1.0")
class BaseBoController(object, metaclass=abc.ABCMeta):
    """The controller module for the Bayesian optimization."""

    @abc.abstractmethod
    def tell(self, study: Study, trials: List[FrozenTrial]) -> None:
        """Tell the observation (trials) to the controller module."""

        raise NotImplementedError

    @abc.abstractmethod
    def ask(self) -> Dict[str, Any]:
        """Ask the next suggested parameters to the controller module."""

        raise NotImplementedError
