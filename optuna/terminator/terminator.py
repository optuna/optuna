import math
from typing import Optional
from typing import Tuple
from typing import Union

from optuna.study.study import Study
from optuna.terminator.improvement.evaluator import BaseImprovementEvaluator
from optuna.terminator.improvement.evaluator import RegretBoundEvaluator
from optuna.terminator.serror import BaseStatisticalErrorEvaluator
from optuna.terminator.serror import CrossValidationStatisticalErrorEvaluator


class Terminator:
    def __init__(
        self,
        regret_bound_evaluator: Optional[BaseImprovementEvaluator] = None,
        statistical_error_evaluator: Optional[BaseStatisticalErrorEvaluator] = None,
    ) -> None:
        self._regret_bound_evaluator = regret_bound_evaluator or RegretBoundEvaluator()
        self._statistical_error_evaluator = (
            statistical_error_evaluator or CrossValidationStatisticalErrorEvaluator()
        )

    def should_terminate(
        self,
        study: Study,
    ) -> Union[bool, Tuple[bool, float, float]]:
        regret_bound = self._regret_bound_evaluator.evaluate(
            trials=study.trials,
            study_direction=study.direction,
        )
        variance = self._statistical_error_evaluator.evaluate(study)
        should_terminate = regret_bound < math.sqrt(variance)

        return should_terminate
