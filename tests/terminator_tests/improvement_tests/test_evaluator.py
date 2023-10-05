from typing import List
from typing import Tuple
from unittest import mock

import numpy as np
import pytest

from optuna.distributions import FloatDistribution
from optuna.study import StudyDirection
from optuna.terminator import BestValueStagnationEvaluator
from optuna.terminator import RegretBoundEvaluator
from optuna.terminator.improvement._preprocessing import NullPreprocessing
from optuna.terminator.improvement.evaluator import BaseImprovementEvaluator
from optuna.terminator.improvement.gp.base import _get_beta
from optuna.terminator.improvement.gp.base import BaseGaussianProcess
from optuna.trial import create_trial
from optuna.trial import FrozenTrial


class _StaticGaussianProcess(BaseGaussianProcess):
    """A dummy BaseGaussianProcess class always returning 0.0 mean and 1.0 std

    This class is introduced to make the GP class and the evaluator class loosely-coupled in unit
    testing.
    """

    def fit(
        self,
        trials: List[FrozenTrial],
    ) -> None:
        pass

    def predict_mean_std(self, trials: List[FrozenTrial]) -> Tuple[np.ndarray, np.ndarray]:
        mean = np.zeros(len(trials))
        std = np.ones(len(trials))

        return mean, std


# TODO(g-votte): test the following edge cases
# TODO(g-votte): - the user specifies non-default top_trials_ratio or min_n_trials


def test_regret_bound_evaluate() -> None:
    trials = [
        create_trial(
            value=0,
            distributions={"a": FloatDistribution(-1.0, 1.0)},
            params={"a": 0.0},
        )
    ]

    # The purpose of the following mock scope is to maintain loose coupling between the tests for
    # preprocessing and those for the `RegretBoundEvaluator` class. The preprocessing logic is
    # thoroughly tested in another file:
    # tests/terminator_tests/improvement_tests/test_preprocessing.py.
    with mock.patch.object(
        RegretBoundEvaluator, "get_preprocessing", return_value=NullPreprocessing()
    ):
        evaluator = RegretBoundEvaluator(gp=_StaticGaussianProcess())
        regret_bound = evaluator.evaluate(trials, study_direction=StudyDirection.MAXIMIZE)
        assert regret_bound == 2.0 * np.sqrt(_get_beta(n_params=1, n_trials=len(trials)))


def test_best_value_stagnation_invalid_argument() -> None:
    with pytest.raises(ValueError):
        # Test that a negative `max_stagnation_trials` raises ValueError.
        BestValueStagnationEvaluator(max_stagnation_trials=-1)


def test_best_value_stagnation_evaluate() -> None:
    evaluator = BestValueStagnationEvaluator(max_stagnation_trials=1)
    # A case of monotonical improvement (best step is the latest element).
    trials = [create_trial(value=value) for value in [0, 1, 2]]
    assert evaluator.evaluate(trials=trials, study_direction=StudyDirection.MAXIMIZE) == 1
    trials = [create_trial(value=value) for value in [2, 1, 0]]
    assert evaluator.evaluate(trials=trials, study_direction=StudyDirection.MINIMIZE) == 1
    # A case of jagged improvement (best step is the second element).
    trials = [create_trial(value=value) for value in [0, 1, 0]]
    assert evaluator.evaluate(trials=trials, study_direction=StudyDirection.MAXIMIZE) == 0
    trials = [create_trial(value=value) for value in [1, 0, 1]]
    assert evaluator.evaluate(trials=trials, study_direction=StudyDirection.MINIMIZE) == 0
    # A case of flat improvement (best step is the first element).
    trials = [create_trial(value=value) for value in [0, 0, 0]]
    assert evaluator.evaluate(trials=trials, study_direction=StudyDirection.MAXIMIZE) == -1
    assert evaluator.evaluate(trials=trials, study_direction=StudyDirection.MINIMIZE) == -1


@pytest.mark.parametrize("evaluator", [RegretBoundEvaluator(), BestValueStagnationEvaluator()])
def test_evaluate_with_no_trial(evaluator: BaseImprovementEvaluator) -> None:
    with pytest.raises(ValueError):
        evaluator.evaluate(trials=[], study_direction=StudyDirection.MAXIMIZE)


def test_evaluate_with_empty_intersection_search_space() -> None:
    evaluator = RegretBoundEvaluator()

    trials = [
        create_trial(
            value=0,
            distributions={},
            params={},
        )
    ]

    with pytest.raises(ValueError):
        evaluator.evaluate(trials=trials, study_direction=StudyDirection.MAXIMIZE)
