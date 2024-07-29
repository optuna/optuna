from __future__ import annotations

import sys

import numpy as np
import pytest

from optuna.distributions import FloatDistribution
from optuna.study.study import create_study
from optuna.terminator import EMMREvaluator
from optuna.terminator import MedianErrorEvaluator
from optuna.trial import create_trial


def test_validation_ratio_to_initial_median_evaluator() -> None:
    with pytest.raises(ValueError):
        MedianErrorEvaluator(paired_improvement_evaluator=EMMREvaluator(), warm_up_trials=-1)

    with pytest.raises(ValueError):
        MedianErrorEvaluator(paired_improvement_evaluator=EMMREvaluator(), n_initial_trials=0)

    with pytest.raises(ValueError):
        MedianErrorEvaluator(paired_improvement_evaluator=EMMREvaluator(), threshold_ratio=0.0)


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
def test_ratio_to_initial_median_evaluator(
    direction: str,
) -> None:
    study = create_study(direction=direction)
    evaluator = MedianErrorEvaluator(
        paired_improvement_evaluator=EMMREvaluator(min_n_trials=3),
        warm_up_trials=1,
        n_initial_trials=3,
        threshold_ratio=2.0,
    )

    study.add_trial(
        create_trial(
            value=1.0,
            distributions={"a": FloatDistribution(-1.0, 10000.0)},
            params={"a": 1.0},
        )
    )
    criterion = evaluator.evaluate(study.trials, study.direction)
    assert criterion == -sys.float_info.max
    study.add_trial(
        create_trial(
            value=22.0,
            distributions={"a": FloatDistribution(-1.0, 10000.0)},
            params={"a": 22.0},
        )
    )
    criterion = evaluator.evaluate(study.trials, study.direction)
    assert criterion == -sys.float_info.max
    study.add_trial(
        create_trial(
            value=333.0,
            distributions={"a": FloatDistribution(-1.0, 10000.0)},
            params={"a": 333.0},
        )
    )
    criterion = evaluator.evaluate(study.trials, study.direction)
    assert criterion == -sys.float_info.max
    study.add_trial(
        create_trial(
            value=4444.0,
            distributions={"a": FloatDistribution(-1.0, 10000.0)},
            params={"a": 4444.0},
        )
    )
    criterion = evaluator.evaluate(study.trials, study.direction)
    assert np.isfinite(criterion)
    assert -sys.float_info.max < criterion
