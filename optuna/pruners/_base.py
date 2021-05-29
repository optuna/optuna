import abc
from typing import Dict

import optuna


class BasePruner(object, metaclass=abc.ABCMeta):
    """Base class for pruners."""

    SPECIAL_KEYWORDS: Dict[str, str] = {}

    def __repr__(self) -> str:
        import inspect

        parameters = []
        for keyword in inspect.signature(self.__class__.__init__).parameters:
            if keyword == "self":
                continue

            access_keyword = "_{keyword}".format(keyword=keyword)
            if keyword in self.SPECIAL_KEYWORDS:
                access_keyword = self.SPECIAL_KEYWORDS[keyword]

            if hasattr(self, access_keyword):
                parameters.append("{}={}".format(keyword, repr(getattr(self, access_keyword))))

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
