from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from optuna.distributions import FloatDistribution
from optuna.study import StudyDirection
from optuna.terminator import EMMREvaluator
from optuna.terminator.improvement.emmr import MARGIN_FOR_NUMARICAL_STABILITY
from optuna.trial import create_trial
from optuna.trial import FrozenTrial


def test_emmr_evaluate() -> None:
    evaluator = EMMREvaluator(min_n_trials=3)
    trials = [
        create_trial(
            value=i,
            distributions={"a": FloatDistribution(-1.0, 10.0)},
            params={"a": float(i)},
        )
        for i in range(3)
    ]
    criterion = evaluator.evaluate(trials=trials, study_direction=StudyDirection.MAXIMIZE)
    assert np.isfinite(criterion)
    assert criterion < sys.float_info.max


def test_emmr_evaluate_with_invalid_argument() -> None:
    with pytest.raises(ValueError):
        EMMREvaluator(min_n_trials=1)


def test_emmr_evaluate_with_insufficient_trials() -> None:
    evaluator = EMMREvaluator()
    trials: list[FrozenTrial] = []
    for _ in range(2):
        criterion = evaluator.evaluate(trials=trials, study_direction=StudyDirection.MAXIMIZE)
        assert math.isclose(criterion, sys.float_info.max * MARGIN_FOR_NUMARICAL_STABILITY)
        trials.append(create_trial(value=0, distributions={}, params={}))


def test_emmr_evaluate_with_empty_intersection_search_space() -> None:
    evaluator = EMMREvaluator()
    trials = [create_trial(value=0, distributions={}, params={}) for _ in range(3)]
    with pytest.warns(UserWarning):
        criterion = evaluator.evaluate(trials=trials, study_direction=StudyDirection.MAXIMIZE)
    assert math.isclose(criterion, sys.float_info.max * MARGIN_FOR_NUMARICAL_STABILITY)
