from __future__ import annotations

from optuna.study._study_direction import StudyDirection
from optuna.study.study import create_study
from optuna.terminator.erroreval import StaticErrorEvaluator
from optuna.terminator.improvement.evaluator import BaseImprovementEvaluator
from optuna.terminator.terminator import Terminator
from optuna.trial import FrozenTrial


class _StaticRegretBoundEvaluator(BaseImprovementEvaluator):
    def __init__(self, constant: float) -> None:
        super().__init__()
        self._constant = constant

    def evaluate(self, trials: list[FrozenTrial], study_direction: StudyDirection) -> float:
        return self._constant


def test_should_terminate() -> None:
    study = create_study()

    # Add dummy trial because Terminator needs at least one trial.
    trial = study.ask()
    study.tell(trial, 0.0)

    # Regret bound is greater than error.
    terminator = Terminator(
        improvement_evaluator=_StaticRegretBoundEvaluator(3),
        error_evaluator=StaticErrorEvaluator(2),
        min_n_trials=1,
    )
    assert not terminator.should_terminate(study)

    # Regret bound is less than error.
    terminator = Terminator(
        improvement_evaluator=_StaticRegretBoundEvaluator(1),
        error_evaluator=StaticErrorEvaluator(2),
        min_n_trials=1,
    )
    assert terminator.should_terminate(study)

    # Regret bound is less than error. However, the number of trials is less than `min_n_trials`.
    terminator = Terminator(
        improvement_evaluator=_StaticRegretBoundEvaluator(1),
        error_evaluator=StaticErrorEvaluator(2),
        min_n_trials=2,
    )
    assert not terminator.should_terminate(study)

    # Regret bound is equal to error.
    terminator = Terminator(
        improvement_evaluator=_StaticRegretBoundEvaluator(2),
        error_evaluator=StaticErrorEvaluator(2),
        min_n_trials=1,
    )
    assert not terminator.should_terminate(study)
