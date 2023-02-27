from typing import Callable
from typing import List

import pytest

from optuna.study.study import create_study
from optuna.terminator.serror import _CROSS_VALIDATION_SCORES_KEY
from optuna.terminator.serror import BaseStatisticalErrorEvaluator
from optuna.terminator.serror import CrossValidationStatisticalErrorEvaluator
from optuna.terminator.serror import report_cross_validation_scores
from optuna.terminator.serror import StaticStatisticalErrorEvaluator
from optuna.trial import create_trial
from optuna.trial import FrozenTrial


def _create_trial(value: float, cv_scores: List[float]) -> FrozenTrial:
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

    evaluator = CrossValidationStatisticalErrorEvaluator()
    serror = evaluator.evaluate(study)

    expected_scale = 1.5
    assert serror == 4.0 * expected_scale


def test_cross_validation_evaluator_without_cv_scores() -> None:
    study = create_study(direction="minimize")
    study.add_trial(
        # Note that the CV score is not reported with the system attr.
        create_trial(params={}, distributions={}, value=0.0)
    )

    evaluator = CrossValidationStatisticalErrorEvaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(study)


def test_report_cross_validation_scores() -> None:
    scores = [1.0, 2.0]

    study = create_study(direction="minimize")
    trial = study.ask()
    report_cross_validation_scores(trial, scores)
    study.tell(trial, 0.0)

    assert study.trials[0].system_attrs[_CROSS_VALIDATION_SCORES_KEY] == scores


def test_static_evaluator() -> None:
    study = create_study(direction="minimize")
    study.add_trials(
        [
            _create_trial(value=2.0, cv_scores=[1.0, -1.0]),
        ]
    )

    evaluator = StaticStatisticalErrorEvaluator(constant=100.0)
    serror = evaluator.evaluate(study)

    assert serror == 100.0


@pytest.mark.parametrize(
    "init_evaluator",
    [
        lambda: CrossValidationStatisticalErrorEvaluator(),
        lambda: StaticStatisticalErrorEvaluator(constant=0.0),
    ],
)
def test_evaluate_with_no_trial(
    init_evaluator: Callable[[], BaseStatisticalErrorEvaluator]
) -> None:
    study = create_study()

    evaluator = init_evaluator()
    with pytest.raises(ValueError):
        evaluator.evaluate(study)
