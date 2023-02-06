from typing import List
from typing import Tuple

import optuna
from optuna.terminator.gp.base import BaseGaussianProcess
from optuna.terminator.regret.evaluator import RegretBoundEvaluator
from optuna.terminator.regret.preprocessing import NullPreprocessing
from optuna.terminator.search_space.intersection import IntersectionSearchSpace


class _StaticGaussianProcess(BaseGaussianProcess):
    """A dummy BaseGaussianProcess class always returning 0.0 mean and 1.0 std

    This class is introduced to make the GP class and the evaluator class loosely-coupled in unit
    testing.
    """

    def fit(
        self,
        trials: List[optuna.trial.FrozenTrial],
    ) -> None:
        self._trials = trials

    def mean_std(
        self,
        trials: List[optuna.trial.FrozenTrial],
    ) -> Tuple[List[float], List[float]]:
        mean = [0.0 for _ in range(len(trials))]
        std = [1.0 for _ in range(len(trials))]
        return mean, std

    def min_ucb(self) -> float:
        return 1.0

    def min_lcb(self, n_additional_candidates: int = 2000) -> float:
        return -1.0

    def gamma(self) -> float:
        space = IntersectionSearchSpace().calculate(self._trials)
        return len(space)

    def t(self) -> float:
        trials = [t for t in self._trials if t.state == optuna.trial.TrialState.COMPLETE]
        return len(trials)


# TODO(g-votte): test the following edge cases
# TODO(g-votte): - multi-objective study
# TODO(g-votte): - no trial has completed, for which the GP class might be responsible
# TODO(g-votte): - trials contain no param, for which the GP class might be responsible
# TODO(g-votte): - study includes inf or -inf objective value
# TODO(g-votte): - both maximize and minimize works (with mock, maybe)
# TODO(g-votte): - the user specifies non-default top_trials_ratio or min_n_trials


def test_evaluate() -> None:
    gp = _StaticGaussianProcess()
    preprocessing = NullPreprocessing()
    evaluator = RegretBoundEvaluator(gp=gp, preprocessing=preprocessing)

    trials = [
        optuna.create_trial(
            value=0,
            distributions={},
            params={},
        )
    ]

    regret_bound = evaluator.evaluate(trials, study_direction=optuna.study.StudyDirection.MAXIMIZE)
    assert regret_bound == 2.0
