import abc
from typing import List
from typing import Optional

import optuna
from optuna.terminator.gp.base import BaseGaussianProcess
from optuna.terminator.gp.gpytorch import GPyTorchGaussianProcess
from optuna.terminator.regret.preprocessing import BasePreprocessing
from optuna.terminator.regret.preprocessing import PreprocessingPipeline
from optuna.terminator.regret.preprocessing import SelectTopTrials
from optuna.terminator.regret.preprocessing import ToMinimize
from optuna.terminator.regret.preprocessing import UnscaleLog


DEFAULT_TOP_TRIALS_RATIO = 0.5
DEFAULT_MIN_N_TRIALS = 20


class BaseRegretBoundEvaluator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(
        self,
        trials: List[optuna.trial.FrozenTrial],
        study_direction: optuna.study.StudyDirection,
    ) -> float:
        pass


class RegretBoundEvaluator:
    def __init__(
        self,
        gp: Optional[BaseGaussianProcess] = None,
        # TODO(g-votte): make top_trials_ratio optional for a non-default preprocessing
        top_trials_ratio: float = DEFAULT_TOP_TRIALS_RATIO,
        # TODO(g-votte): make min_n_trials optional for a non-default preprocessing
        min_n_trials: int = DEFAULT_MIN_N_TRIALS,
        preprocessing: Optional[BasePreprocessing] = None,
    ) -> None:
        self._gp = gp or GPyTorchGaussianProcess()
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
            ]
        )

    def evaluate(
        self,
        trials: List[optuna.trial.FrozenTrial],
        study_direction: optuna.study.StudyDirection,
    ) -> float:
        trials = self._preprocessing.apply(trials, study_direction)
        self._gp.fit(trials)
        regret_bound = self._gp.min_ucb() - self._gp.min_lcb()

        return regret_bound
