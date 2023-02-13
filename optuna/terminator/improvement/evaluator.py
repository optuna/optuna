import abc
from typing import List
from typing import Optional

import optuna
from optuna.terminator.gp.base import BaseMinUcbLcbEstimator
from optuna.terminator.gp.botorch import BoTorchMinUcbLcbEstimator
from optuna.terminator.improvement.preprocessing import BasePreprocessing
from optuna.terminator.improvement.preprocessing import OneToHot
from optuna.terminator.improvement.preprocessing import PreprocessingPipeline
from optuna.terminator.improvement.preprocessing import SelectTopTrials
from optuna.terminator.improvement.preprocessing import ToIntersectionSearchSpace
from optuna.terminator.improvement.preprocessing import ToMinimize
from optuna.terminator.improvement.preprocessing import UnscaleLog


DEFAULT_TOP_TRIALS_RATIO = 0.5
DEFAULT_MIN_N_TRIALS = 20


class BaseImprovementEvaluator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(
        self,
        trials: List[optuna.trial.FrozenTrial],
        study_direction: optuna.study.StudyDirection,
    ) -> float:
        pass


class RegretBoundEvaluator(BaseImprovementEvaluator):
    def __init__(
        self,
        estimator: Optional[BaseMinUcbLcbEstimator] = None,
        # TODO(g-votte): make top_trials_ratio optional for a non-default preprocessing
        top_trials_ratio: float = DEFAULT_TOP_TRIALS_RATIO,
        # TODO(g-votte): make min_n_trials optional for a non-default preprocessing
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
        preprocessing: Optional[BasePreprocessing] = None,
    ) -> None:
        self._estimator = estimator or BoTorchMinUcbLcbEstimator()
        self._top_trials_ratio = top_trials_ratio
        self._min_n_trials = min_n_trials
        self._preprocessing = preprocessing or PreprocessingPipeline(
            [
                SelectTopTrials(
                    top_trials_ratio=self._top_trials_ratio,
                    min_n_trials=self._min_n_trials,
                ),
                UnscaleLog(),
                ToMinimize(),
                ToIntersectionSearchSpace(),
                OneToHot(),
            ]
        )

    def evaluate(
        self,
        trials: List[optuna.trial.FrozenTrial],
        study_direction: optuna.study.StudyDirection,
    ) -> float:
        trials = self._preprocessing.apply(trials, study_direction)
        self._estimator.fit(trials)
        regret_bound = self._estimator.min_ucb() - self._estimator.min_lcb()

        return regret_bound
