from __future__ import annotations

import pytest

from optuna.study._study_direction import StudyDirection
from optuna.study.study import create_study
from optuna.terminator import BaseImprovementEvaluator
from optuna.terminator import StaticErrorEvaluator
from optuna.terminator import Terminator
from optuna.terminator.improvement.evaluator import BestValueStagnationEvaluator
from optuna.trial import FrozenTrial


class _StaticImprovementEvaluator(BaseImprovementEvaluator):
    def __init__(self, constant: float) -> None:
        super().__init__()
        self._constant = constant

    def evaluate(self, trials: list[FrozenTrial], study_direction: StudyDirection) -> float:
        return self._constant


def test_init() -> None:
    # Test that a positive `min_n_trials` does not raise any error.
    Terminator(min_n_trials=1)

    with pytest.raises(ValueError):
        # Test that a non-positive `min_n_trials` raises ValueError.
        Terminator(min_n_trials=0)

    Terminator(BestValueStagnationEvaluator(), min_n_trials=1)


def test_should_terminate() -> None:
    study = create_study()

    # Add dummy trial because Terminator needs at least one trial.
    trial = study.ask()
    study.tell(trial, 0.0)

    # Improvement is greater than error.
    terminator = Terminator(
        improvement_evaluator=_StaticImprovementEvaluator(3),
        error_evaluator=StaticErrorEvaluator(0),
        min_n_trials=1,
    )
    assert not terminator.should_terminate(study)

    # Improvement is less than error.
    terminator = Terminator(
        improvement_evaluator=_StaticImprovementEvaluator(-1),
        error_evaluator=StaticErrorEvaluator(0),
        min_n_trials=1,
    )
    assert terminator.should_terminate(study)

    # Improvement is less than error. However, the number of trials is less than `min_n_trials`.
    terminator = Terminator(
        improvement_evaluator=_StaticImprovementEvaluator(-1),
        error_evaluator=StaticErrorEvaluator(0),
        min_n_trials=2,
    )
    assert not terminator.should_terminate(study)

    # Improvement is equal to error.
    terminator = Terminator(
        improvement_evaluator=_StaticImprovementEvaluator(0),
        error_evaluator=StaticErrorEvaluator(0),
        min_n_trials=1,
    )
    assert not terminator.should_terminate(study)
