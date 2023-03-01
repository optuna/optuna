import abc
from typing import List

import numpy as np

from optuna._experimental import experimental_class
from optuna.study.study import Study
from optuna.trial import Trial
from optuna.trial._state import TrialState


_CROSS_VALIDATION_SCORES_KEY = "terminator:cv_scores"


class BaseStatisticalErrorEvaluator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(
        self,
        study: Study,
    ) -> float:
        pass


@experimental_class("3.2.0")
class CrossValidationStatisticalErrorEvaluator(BaseStatisticalErrorEvaluator):
    def evaluate(
        self,
        study: Study,
    ) -> float:
        assert len(study.get_trials(states=(TrialState.COMPLETE,))) != 0

        best_trial_attrs = study.best_trial.system_attrs
        if _CROSS_VALIDATION_SCORES_KEY in best_trial_attrs:
            cv_scores = best_trial_attrs[_CROSS_VALIDATION_SCORES_KEY]
        else:
            raise ValueError(
                "Cross-validation scores have not been reported. Please call "
                "`report_cross_validation_scores(trial, scores)` during a trial and pass the "
                "list of scores as `scores`."
            )

        k = len(cv_scores)
        scale = 1 / k + 1 / (k - 1)

        var = scale * np.var(cv_scores)

        return float(var)


@experimental_class("3.2.0")
def report_cross_validation_scores(trial: Trial, scores: List[float]) -> None:
    if len(scores) <= 1:
        raise ValueError("The length of `scores` is expected to be greater than one.")
    trial.storage.set_trial_system_attr(trial._trial_id, _CROSS_VALIDATION_SCORES_KEY, scores)


@experimental_class("3.2.0")
class StaticStatisticalErrorEvaluator(BaseStatisticalErrorEvaluator):
    def __init__(self, constant: float) -> None:
        self._constant = constant

    def evaluate(
        self,
        study: Study,
    ) -> float:
        return self._constant
