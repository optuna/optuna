from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type

import numpy as np

import optuna
from optuna._batch_study import BatchStudy
from optuna.multi_objective.study import MultiObjectiveStudy
from optuna.multi_objective.trial import BatchMultiObjectiveTrial
from optuna.multi_objective.trial import FrozenMultiObjectiveTrial
from optuna.trial import BatchTrial

BatchMultiObjectiveFuncType = Callable[[BatchMultiObjectiveTrial], Sequence[np.ndarray]]
MultiObjectiveCallbackFuncType = Callable[
    [
        "optuna.multi_objective.study.MultiObjectiveStudy",
        "optuna.multi_objective.trial.FrozenMultiObjectiveTrial",
    ],
    None,
]


class BatchMultiObjectiveStudy(MultiObjectiveStudy):
    def __init__(self, study: MultiObjectiveStudy, batch_size: int) -> None:
        self._sampler = study.sampler
        super().__init__(BatchStudy(study, batch_size))

    @property
    def sampler(self) -> "optuna.multi_objective.samplers.BaseMultiObjectiveSampler":
        return self._sampler

    def batch_optimize(
        self,
        objective: BatchMultiObjectiveFuncType,
        timeout: Optional[int] = None,
        n_batches: Optional[int] = None,
        n_jobs: int = 1,
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[MultiObjectiveCallbackFuncType]] = None,
        gc_after_trial: bool = True,
        show_progress_bar: bool = False,
    ) -> None:
        def mo_objective(trial: BatchTrial) -> np.ndarray:
            trials = [
                optuna.multi_objective.trial.MultiObjectiveTrial(trial) for trial in trial._trials
            ]
            mo_trial = BatchMultiObjectiveTrial(trials)
            values = objective(mo_trial)
            mo_trial._report_complete_values(values)
            return np.zeros(len(trials))  # Dummy value.

        # Wraps a multi-objective callback so that we can pass it to the `Study.optimize` method.
        def wrap_mo_callback(
            callback: MultiObjectiveCallbackFuncType,
        ) -> Callable[["optuna.Study", "optuna.trial.FrozenTrial"], None]:
            return lambda study, trial: callback(
                MultiObjectiveStudy(study), FrozenMultiObjectiveTrial(self.n_objectives, trial),
            )

        if callbacks is None:
            wrapped_callbacks = None
        else:
            wrapped_callbacks = [wrap_mo_callback(callback) for callback in callbacks]

        assert isinstance(self._study, BatchStudy)
        self._study.batch_optimize(
            mo_objective,
            timeout=timeout,
            n_batches=n_batches,
            n_jobs=n_jobs,
            catch=catch,
            callbacks=wrapped_callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
        )
