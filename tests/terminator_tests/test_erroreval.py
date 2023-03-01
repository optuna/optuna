from __future__ import annotations

import math

import pytest

from optuna.study.study import create_study
from optuna.terminator.erroreval import _CROSS_VALIDATION_SCORES_KEY
from optuna.terminator.erroreval import CrossValidationErrorEvaluator
from optuna.terminator.erroreval import report_cross_validation_scores
from optuna.terminator.erroreval import StaticErrorEvaluator
from optuna.trial import create_trial
from optuna.trial import FrozenTrial


def _create_trial(value: float, cv_scores: list[float]) -> FrozenTrial:
    return create_trial(
        params={},
        distributions={},
        value=value,
        system_attrs={_CROSS_VALIDATION_SCORES_KEY: cv_scores},
    )


def test_cross_validation_evaluator() -> None:
    study = create_study(direction="minimize")
    study.add_trials(
        [
            _create_trial(value=2.0, cv_scores=[1.0, -1.0]),  # second best trial with 1.0 var
            _create_trial(value=1.0, cv_scores=[2.0, -2.0]),  # best trial with 4.0 var
        ]
    )

    evaluator = CrossValidationErrorEvaluator()
    serror = evaluator.evaluate(study)

    expected_scale = 1.5
    assert serror == math.sqrt(4.0 * expected_scale)


def test_cross_validation_evaluator_without_cv_scores() -> None:
    study = create_study(direction="minimize")
    study.add_trial(
        # Note that the CV score is not reported with the system attr.
        create_trial(params={}, distributions={}, value=0.0)
    )

    evaluator = CrossValidationErrorEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(study)


def test_report_cross_validation_scores() -> None:
    scores = [1.0, 2.0]

    study = create_study(direction="minimize")
    trial = study.ask()
    report_cross_validation_scores(trial, scores)
    study.tell(trial, 0.0)

    assert study.trials[0].system_attrs[_CROSS_VALIDATION_SCORES_KEY] == scores


def test_report_cross_validation_scores_with_illegal_scores_length() -> None:
    scores = [1.0]

    study = create_study(direction="minimize")
    trial = study.ask()
    with pytest.raises(ValueError):
        report_cross_validation_scores(trial, scores)


def test_static_evaluator() -> None:
    study = create_study(direction="minimize")
    study.add_trials(
        [
            _create_trial(value=2.0, cv_scores=[1.0, -1.0]),
        ]
    )

    evaluator = StaticErrorEvaluator(constant=100.0)
    serror = evaluator.evaluate(study)

    assert serror == 100.0
