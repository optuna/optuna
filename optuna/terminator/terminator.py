import abc
from typing import Optional

from optuna._experimental import experimental_class
from optuna.study.study import Study
from optuna.terminator.erroreval import BaseErrorEvaluator
from optuna.terminator.erroreval import CrossValidationErrorEvaluator
from optuna.terminator.improvement.evaluator import BaseImprovementEvaluator
from optuna.terminator.improvement.evaluator import DEFAULT_MIN_N_TRIALS
from optuna.terminator.improvement.evaluator import RegretBoundEvaluator
from optuna.trial import TrialState


class BaseTerminator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def should_terminate(self, study: Study) -> bool:
        pass


@experimental_class("3.2.0")
class Terminator(BaseTerminator):
    def __init__(
        self,
        improvement_evaluator: Optional[BaseImprovementEvaluator] = None,
        error_evaluator: Optional[BaseErrorEvaluator] = None,
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
    ) -> None:
        self._improvement_evaluator = improvement_evaluator or RegretBoundEvaluator()
        self._error_evaluator = error_evaluator or CrossValidationErrorEvaluator()
        self._min_n_trials = min_n_trials

    def should_terminate(self, study: Study) -> bool:
        trials = study.get_trials(states=[TrialState.COMPLETE])

        if len(trials) < self._min_n_trials:
            return False

        regret_bound = self._improvement_evaluator.evaluate(
            trials=study.trials,
            study_direction=study.direction,
        )
        error = self._error_evaluator.evaluate(study)
        should_terminate = regret_bound < error

        return should_terminate
