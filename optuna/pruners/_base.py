import abc
from typing import Any
from typing import Dict
import warnings

import optuna


class BasePruner(object, metaclass=abc.ABCMeta):
    """Base class for pruners."""

    def __repr__(self) -> str:
        parameters = []
        for key, value in self._get_init_arguments().items():
            parameters.append("{}={}".format(key, repr(value)))

        return "{class_name}({parameters})".format(
            class_name=self.__class__.__name__,
            parameters=",".join(parameters),
        )

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

    def _get_init_arguments(self) -> Dict[str, Any]:
        warnings.warn(
            "{} doesn't have a method `_get_init_arguments`."
            " If you implement a pruner in your own, consider implementing it."
            " For more information, please refer to Optuna GitHub PR"
            " (https://github.com/optuna/optuna/pull/2707).".format(self.__class__.__name__)
        )
        return {}
