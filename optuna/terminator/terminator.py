import abc
import math
from typing import Optional

from optuna._experimental import experimental_class
from optuna.study.study import Study
from optuna.terminator.improvement.evaluator import BaseImprovementEvaluator
from optuna.terminator.improvement.evaluator import DEFAULT_MIN_N_TRIALS
from optuna.terminator.improvement.evaluator import RegretBoundEvaluator
from optuna.terminator.serror import BaseStatisticalErrorEvaluator
from optuna.terminator.serror import CrossValidationStatisticalErrorEvaluator
from optuna.trial import TrialState


class BaseTerminator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def should_terminate(self, study: Study) -> bool:
        pass


@experimental_class("3.2.0")
class Terminator(BaseTerminator):
    def __init__(
        self,
        regret_bound_evaluator: Optional[BaseImprovementEvaluator] = None,
        statistical_error_evaluator: Optional[BaseStatisticalErrorEvaluator] = None,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
    ) -> None:
        self._regret_bound_evaluator = regret_bound_evaluator or RegretBoundEvaluator()
        self._statistical_error_evaluator = (
            statistical_error_evaluator or CrossValidationStatisticalErrorEvaluator()
        )
        self._min_n_trials = min_n_trials

    def should_terminate(self, study: Study) -> bool:
        trials = study.get_trials(states=[TrialState.COMPLETE])

        if len(trials) < self._min_n_trials:
            return False

        regret_bound = self._regret_bound_evaluator.evaluate(
            trials=study.trials,
            study_direction=study.direction,
        )
        variance = self._statistical_error_evaluator.evaluate(study)
        should_terminate = regret_bound < math.sqrt(variance)

        return should_terminate
