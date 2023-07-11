from __future__ import annotations

import math

import pytest

from optuna.study.study import create_study
from optuna.terminator import CrossValidationErrorEvaluator
from optuna.terminator import report_cross_validation_scores
from optuna.terminator import StaticErrorEvaluator
from optuna.terminator.erroreval import _CROSS_VALIDATION_SCORES_KEY
from optuna.trial import create_trial
from optuna.trial import FrozenTrial


def _create_trial(value: float, cv_scores: list[float]) -> FrozenTrial:
    return create_trial(
        params={},
        distributions={},
        value=value,
        system_attrs={_CROSS_VALIDATION_SCORES_KEY: cv_scores},
    )


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_cross_validation_evaluator(direction: str) -> None:
    study = create_study(direction=direction)
    sign = 1 if direction == "minimize" else -1
    study.add_trials(
        [
            _create_trial(
                value=sign * 2.0, cv_scores=[1.0, -1.0]
            ),  # Second best trial with 1.0 var.
            _create_trial(value=sign * 1.0, cv_scores=[2.0, -2.0]),  # Best trial with 4.0 var.
        ]
    )

    evaluator = CrossValidationErrorEvaluator()
    serror = evaluator.evaluate(study.trials, study.direction)

    expected_scale = 1.5
    assert serror == math.sqrt(4.0 * expected_scale)


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_cross_validation_evaluator_without_cv_scores(direction: str) -> None:
    study = create_study(direction=direction)
    study.add_trial(
        # Note that the CV score is not reported with the system attr.
        create_trial(params={}, distributions={}, value=0.0)
    )

    evaluator = CrossValidationErrorEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(study.trials, study.direction)


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_report_cross_validation_scores(direction: str) -> None:
    scores = [1.0, 2.0]

    study = create_study(direction=direction)
    trial = study.ask()
    report_cross_validation_scores(trial, scores)
    study.tell(trial, 0.0)

    assert study.trials[0].system_attrs[_CROSS_VALIDATION_SCORES_KEY] == scores


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_report_cross_validation_scores_with_illegal_scores_length(direction: str) -> None:
    scores = [1.0]

    study = create_study(direction=direction)
    trial = study.ask()
    with pytest.raises(ValueError):
        report_cross_validation_scores(trial, scores)


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_static_evaluator(direction: str) -> None:
    study = create_study(direction=direction)
    study.add_trials(
        [
            _create_trial(value=2.0, cv_scores=[1.0, -1.0]),
        ]
    )

    evaluator = StaticErrorEvaluator(constant=100.0)
    serror = evaluator.evaluate(study.trials, study.direction)

    assert serror == 100.0
