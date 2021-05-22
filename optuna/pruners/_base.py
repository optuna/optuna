import abc
from typing import List

import optuna


class BasePruner(object, metaclass=abc.ABCMeta):
    """Base class for pruners."""

    @abc.abstractmethod
    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        """Judge whether the trial should be pruned based on the reported values.

        Note that this method is not supposed to be called by library users. Instead,
        :func:`optuna.trial.Trial.report` and :func:`optuna.trial.Trial.should_prune` provide
        user interfaces to implement pruning mechanism in an objective function.

        Args:
            study:
                Study object of the target study.
            trial:
                FrozenTrial object of the target trial.
                Take a copy before modifying this object.

        Returns:
            A boolean value representing whether the trial should be pruned.
        """

        raise NotImplementedError

    def _arguments(self) -> List[str]:
        raise []

    def __repr__(self) -> str:
        arguments = ", ".join(self._arguments())
        return f"{self.__class__.__name__}({arguments})"
