import abc
import math
from typing import Optional

from optuna.study.study import Study
from optuna.terminator.improvement.evaluator import BaseImprovementEvaluator
from optuna.terminator.improvement.evaluator import RegretBoundEvaluator
from optuna.terminator.serror import BaseStatisticalErrorEvaluator
from optuna.terminator.serror import CrossValidationStatisticalErrorEvaluator


class BaseTerminator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def should_terminate(self, study: Study) -> bool:
        pass


class Terminator(BaseTerminator):
    def __init__(
        self,
        regret_bound_evaluator: Optional[BaseImprovementEvaluator] = None,
        statistical_error_evaluator: Optional[BaseStatisticalErrorEvaluator] = None,
    ) -> None:
        self._regret_bound_evaluator = regret_bound_evaluator or RegretBoundEvaluator()
        self._statistical_error_evaluator = (
            statistical_error_evaluator or CrossValidationStatisticalErrorEvaluator()
        )

    def should_terminate(self, study: Study) -> bool:
        regret_bound = self._regret_bound_evaluator.evaluate(
            trials=study.trials,
            study_direction=study.direction,
        )
        variance = self._statistical_error_evaluator.evaluate(study)
        should_terminate = regret_bound < math.sqrt(variance)

        return should_terminate
