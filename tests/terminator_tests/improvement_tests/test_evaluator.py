from typing import List

import optuna
from optuna.terminator.gp.base import BaseMinUcbLcbEstimator
from optuna.terminator.improvement.evaluator import RegretBoundEvaluator
from optuna.terminator.improvement.preprocessing import NullPreprocessing


class _StaticMinUcbLcbEstimator(BaseMinUcbLcbEstimator):
    """A dummy BaseGaussianProcess class always returning 0.0 mean and 1.0 std

    This class is introduced to make the GP class and the evaluator class loosely-coupled in unit
    testing.
    """

    def fit(
        self,
        trials: List[optuna.trial.FrozenTrial],
    ) -> None:
        self._trials = trials

    def min_ucb(self) -> float:
        return 1.0

    def min_lcb(self) -> float:
        return -1.0


# TODO(g-votte): test the following edge cases
# TODO(g-votte): - multi-objective study
# TODO(g-votte): - no trial has completed, for which the GP class might be responsible
# TODO(g-votte): - trials contain no param, for which the GP class might be responsible
# TODO(g-votte): - study includes inf or -inf objective value
# TODO(g-votte): - both maximize and minimize works (with mock, maybe)
# TODO(g-votte): - the user specifies non-default top_trials_ratio or min_n_trials


def test_evaluate() -> None:
    estimator = _StaticMinUcbLcbEstimator()
    preprocessing = NullPreprocessing()
    evaluator = RegretBoundEvaluator(estimator=estimator, preprocessing=preprocessing)

    trials = [
        optuna.create_trial(
            value=0,
            distributions={},
            params={},
        )
    ]

    regret_bound = evaluator.evaluate(trials, study_direction=optuna.study.StudyDirection.MAXIMIZE)
    assert regret_bound == 2.0
