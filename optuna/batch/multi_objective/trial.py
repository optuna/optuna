from typing import Sequence

import numpy as np

import optuna
from optuna.batch.trial import BaseBatchTrial


class BatchMultiObjectiveTrial(BaseBatchTrial):
    def __init__(
        self, trials: Sequence["optuna.multi_objective.trial.MultiObjectiveTrial"]
    ) -> None:
        self._trials = trials

    def _get_trials(self) -> Sequence["optuna.multi_objective.trial.MultiObjectiveTrial"]:
        return self._trials

    def report(self, values: Sequence[np.ndarray], step: int) -> None:
        values = np.array(values).transpose()
        for value, trial in zip(values, self._trials):
            trial.report(value, step)

    def _report_complete_values(self, values: Sequence[np.ndarray]) -> None:
        values = np.array(values).transpose()
        for value, trial in zip(values, self._trials):
            trial._report_complete_values(value)
