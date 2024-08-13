from __future__ import annotations

import numpy as np
import pytest

from optuna.distributions import FloatDistribution
from optuna.study import StudyDirection
from optuna.terminator import EMMREvaluator
from optuna.terminator import MedianErrorEvaluator
from optuna.trial import create_trial
from optuna.trial import FrozenTrial


def test_validation_ratio_to_initial_median_evaluator() -> None:
    with pytest.raises(ValueError):
        MedianErrorEvaluator(paired_improvement_evaluator=EMMREvaluator(), warm_up_trials=-1)

    with pytest.raises(ValueError):
        MedianErrorEvaluator(paired_improvement_evaluator=EMMREvaluator(), n_initial_trials=0)

    with pytest.raises(ValueError):
        MedianErrorEvaluator(paired_improvement_evaluator=EMMREvaluator(), threshold_ratio=0.0)


@pytest.mark.parametrize("direction", [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
def test_ratio_to_initial_median_evaluator(
    direction: StudyDirection,
) -> None:
    trials: list[FrozenTrial] = []
    evaluator = MedianErrorEvaluator(
        paired_improvement_evaluator=EMMREvaluator(min_n_trials=3),
        warm_up_trials=1,
        n_initial_trials=3,
        threshold_ratio=2.0,
    )

    trials.append(
        create_trial(
            value=1.0,
            distributions={"a": FloatDistribution(-1.0, 10000.0)},
            params={"a": 1.0},
        )
    )
    criterion = evaluator.evaluate(trials, direction)
    assert criterion <= 0.0
    trials.append(
        create_trial(
            value=22.0,
            distributions={"a": FloatDistribution(-1.0, 10000.0)},
            params={"a": 22.0},
        )
    )
    criterion = evaluator.evaluate(trials, direction)
    assert criterion <= 0.0
    trials.append(
        create_trial(
            value=333.0,
            distributions={"a": FloatDistribution(-1.0, 10000.0)},
            params={"a": 333.0},
        )
    )
    criterion = evaluator.evaluate(trials, direction)
    assert criterion <= 0.0
    trials.append(
        create_trial(
            value=4444.0,
            distributions={"a": FloatDistribution(-1.0, 10000.0)},
            params={"a": 4444.0},
        )
    )
    criterion = evaluator.evaluate(trials, direction)
    assert np.isfinite(criterion)
    assert 0.0 <= criterion
