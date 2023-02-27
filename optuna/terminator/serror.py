import abc

import numpy as np

from optuna._experimental import experimental_class
from optuna.study.study import Study
from optuna.trial._state import TrialState


class BaseStatisticalErrorEvaluator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(
        self,
        study: Study,
    ) -> float:
        pass


@experimental_class("3.2.0")
class CrossValidationStatisticalErrorEvaluator(BaseStatisticalErrorEvaluator):
    def __init__(self, user_attr_key: str = "cv_scores") -> None:
        self._user_attr_key = user_attr_key

    def evaluate(
        self,
        study: Study,
    ) -> float:
        if len(study.get_trials(states=(TrialState.COMPLETE,))) == 0:
            raise ValueError(
                "Statistical error cannot be calculated because no trial has been completed."
            )

        cv_scores = study.best_trial.user_attrs[self._user_attr_key]

        k = len(cv_scores)
        scale = 1 / k + 1 / (k - 1)

        var = scale * np.var(cv_scores)

        return float(var)


@experimental_class("3.2.0")
class StaticStatisticalErrorEvaluator(BaseStatisticalErrorEvaluator):
    def __init__(self, constant: float) -> None:
        self._constant = constant

    def evaluate(
        self,
        study: Study,
    ) -> float:
        if len(study.get_trials(states=(TrialState.COMPLETE,))) == 0:
            raise ValueError(
                "Statistical error cannot be calculated because no trial has been completed."
            )

        return self._constant
