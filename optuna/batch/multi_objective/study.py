from functools import partial
import math
import types
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np

import optuna
from optuna._experimental import experimental
from optuna.batch.study import _run_trial_and_callbacks

ObjectiveFuncType = Callable[
    ["optuna.batch.multi_objective.trial.BatchMultiObjectiveTrial"], Sequence[np.ndarray]
]
CallbackFuncType = Callable[
    [
        "optuna.multi_objective.study.MultiObjectiveStudy",
        "optuna.multi_objective.trial.FrozenMultiObjectiveTrial",
    ],
    None,
]


class BatchMultiObjectiveStudy(object):
    def __init__(self, study: "optuna.multi_objective.study.MultiObjectiveStudy"):
        self._study = study

    def __getattr__(self, attr_name: str) -> Any:
        return getattr(self._study, attr_name)

    def optimize(
        self,
        objective: ObjectiveFuncType,
        timeout: Optional[int] = None,
        n_trials: Optional[int] = None,
        batch_size: int = 1,
        n_jobs: int = 1,
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[CallbackFuncType]] = None,
        gc_after_trial: bool = True,
        show_progress_bar: bool = False,
    ) -> None:
        def mo_objective(trial: optuna.batch.trial.BatchTrial) -> np.ndarray:
            trials = [
                optuna.multi_objective.trial.MultiObjectiveTrial(trial) for trial in trial._trials
            ]
            mo_trial = optuna.batch.multi_objective.trial.BatchMultiObjectiveTrial(trials)
            values = objective(mo_trial)
            mo_trial._report_complete_values(values)
            return np.zeros(len(trials))  # Dummy value.

        self._study._study._run_trial_and_callbacks = types.MethodType(  # type: ignore
            partial(_run_trial_and_callbacks, batch_func=mo_objective, batch_size=batch_size),
            self._study._study,
        )

        n_trials = math.ceil(n_trials / batch_size) if n_trials is not None else None

        self._study.optimize(
            lambda _: [0] * self._study.n_objectives,
            timeout=timeout,
            n_trials=n_trials,
            n_jobs=n_jobs,
            catch=catch,
            callbacks=callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
        )


@experimental("2.1.0")
def create_study(
    directions: List[str],
    study_name: Optional[str] = None,
    storage: Optional[Union[str, "optuna.storages.BaseStorage"]] = None,
    sampler: Optional["optuna.multi_objective.samplers.BaseMultiObjectiveSampler"] = None,
    load_if_exists: bool = False,
) -> BatchMultiObjectiveStudy:

    study = optuna.multi_objective.create_study(
        directions, study_name, storage, sampler, load_if_exists
    )
    return BatchMultiObjectiveStudy(study)


@experimental("2.1.0")
def load_study(
    study_name: str,
    storage: Optional[Union[str, "optuna.storages.BaseStorage"]] = None,
    sampler: Optional["optuna.multi_objective.samplers.BaseMultiObjectiveSampler"] = None,
) -> BatchMultiObjectiveStudy:

    study = optuna.multi_objective.load_study(study_name, storage, sampler)
    return BatchMultiObjectiveStudy(study)
