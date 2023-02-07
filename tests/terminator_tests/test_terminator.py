from typing import List

from optuna.study._study_direction import StudyDirection
from optuna.study.study import create_study
from optuna.terminator.regret.evaluator import BaseRegretBoundEvaluator
from optuna.terminator.serror import StaticStatisticalErrorEvaluator
from optuna.terminator.terminator import Terminator
from optuna.trial import FrozenTrial


class _StaticRegretBoundEvaluator(BaseRegretBoundEvaluator):
    def __init__(self, constant: float) -> None:
        super().__init__()
        self._constant = constant

    def evaluate(self, trials: List[FrozenTrial], study_direction: StudyDirection) -> float:
        return self._constant


def test_should_terminate() -> None:
    study = create_study()

    # Add dummy trial because Terminator needs at least one trial.
    trial = study.ask()
    study.tell(trial, 0.0)

    # Regret bound is greater than sqrt of variance.
    terminator = Terminator(
        regret_bound_evaluator=_StaticRegretBoundEvaluator(11),
        statistical_error_evaluator=StaticStatisticalErrorEvaluator(100),
    )
    assert not terminator.should_terminate(study)

    # Regret bound is less than sqrt of variance.
    terminator = Terminator(
        regret_bound_evaluator=_StaticRegretBoundEvaluator(9),
        statistical_error_evaluator=StaticStatisticalErrorEvaluator(100),
    )
    assert terminator.should_terminate(study)

    # Regret bound is equal to sqrt of variance.
    terminator = Terminator(
        regret_bound_evaluator=_StaticRegretBoundEvaluator(10),
        statistical_error_evaluator=StaticStatisticalErrorEvaluator(100),
    )
    assert not terminator.should_terminate(study)
