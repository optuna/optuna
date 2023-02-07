from typing import Callable
from typing import List

import pytest

from optuna.study.study import create_study
from optuna.terminator.serror import BaseStatisticalErrorEvaluator
from optuna.terminator.serror import CrossValidationStatisticalErrorEvaluator
from optuna.terminator.serror import StaticStatisticalErrorEvaluator
from optuna.trial import create_trial
from optuna.trial import FrozenTrial


def _create_trial(value: float, cv_scores: List[float]) -> FrozenTrial:
    return create_trial(
        params={}, distributions={}, value=value, user_attrs={"cv_scores": cv_scores}
    )


def test_cross_validation() -> None:
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


def test_static() -> None:
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
