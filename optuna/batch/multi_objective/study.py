import math
import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np

import optuna
from optuna._experimental import experimental
from optuna.batch.study import _BaseObjectiveCallbackWrapper
import optuna.logging
from optuna.multi_objective.trial import FrozenMultiObjectiveTrial

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

_logger = optuna.logging.get_logger(__name__)


class _ObjectiveCallbackWrapper(_BaseObjectiveCallbackWrapper):
    def __init__(
        self,
        study: "optuna.multi_objective.study.MultiObjectiveStudy",
        objective: ObjectiveFuncType,
        batch_size: int,
    ):
        self._study = study
        self._batch_size = batch_size
        self._objective = objective
        self._members = {}  # type: Dict[int, List[int]]

    def batch_objective(
        self, trial: "optuna.multi_objective.trial.MultiObjectiveTrial"
    ) -> Sequence[float]:
        # Assume storage has already been synchronized.
        _new_so_trials = self._create_trials(self._study._study, trial._trial, self._batch_size)
        self._members[trial._trial._trial_id] = [t._trial_id for t in _new_so_trials]
        trials = [trial] + [
            optuna.multi_objective.trial.MultiObjectiveTrial(t) for t in _new_so_trials
        ]
        batch_trial = optuna.batch.multi_objective.trial.BatchMultiObjectiveTrial(trials)
        try:
            results = self._objective(batch_trial)
            transposed_results = np.array(results).transpose()
        except Exception as e:
            for trial in trials:
                message = "Trial {} failed because of the following error: {}".format(
                    trial.number, repr(e)
                )
                _logger.warning(message, exc_info=True)
                trial_id = trial._trial._trial_id
                self._study._storage.set_trial_system_attr(trial_id, "fail_reason", message)
                self._study._storage.set_trial_state(trial_id, optuna.trial.TrialState.FAIL)
            raise

        for trial, result in zip(trials, transposed_results):
            trial_id = trial._trial._trial_id
            trial._report_complete_values(result)
            _logger.info(
                "Trial {} finished with values: {} with parameters: {}.".format(
                    trial._trial.number, result, trial._trial.params
                )
            )
            # Set dummy objective value.
            self._study._storage.set_trial_value(trial_id, 0)
            self._study._storage.set_trial_state(trial_id, optuna.trial.TrialState.COMPLETE)
        return transposed_results[0]

    def wrap_callback(self, callback: CallbackFuncType) -> CallbackFuncType:
        def _callback(
            study: "optuna.multi_objective.study.MultiObjectiveStudy",
            trial: "optuna.multi_objective.trial.FrozenMultiObjectiveTrial",
        ) -> None:
            callback(study, trial)
            for member_id in self._members[trial._trial_id]:
                _trial = study._study._storage.get_trial(member_id)
                mo_trial = FrozenMultiObjectiveTrial(study.n_objectives, _trial)
                callback(study, mo_trial)

        return _callback


class BatchMultiObjectiveStudy(object):
    def __init__(self, study: "optuna.multi_objective.study.MultiObjectiveStudy", batch_size: int):
        self._study = study
        self._batch_size = batch_size

    def __getattr__(self, attr_name: str) -> Any:
        return getattr(self._study, attr_name)

    def optimize(
        self,
        objective: ObjectiveFuncType,
        timeout: Optional[int] = None,
        n_batches: Optional[int] = None,
        n_jobs: int = 1,
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[CallbackFuncType]] = None,
        gc_after_trial: bool = True,
        show_progress_bar: bool = False,
    ) -> None:

        wrapper = _ObjectiveCallbackWrapper(self._study, objective, self._batch_size)

        if callbacks is None:
            wrapped_callbacks = None
        else:
            wrapped_callbacks = [wrapper.wrap_callback(callback) for callback in callbacks]

        n_trials = math.ceil(n_batches / n_jobs) if n_batches is not None else None

        self._study._study._run_trial = types.MethodType(  # type: ignore
            _run_trial,
            self._study._study,
        )
        self._study.optimize(
            wrapper.batch_objective,
            timeout=timeout,
            n_trials=n_trials,
            n_jobs=n_jobs,
            catch=catch,
            callbacks=wrapped_callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
        )

    @property
    def batch_size(self) -> int:
        """Return the size of batches.

        Returns:
            Size of batches.
        """

        return self._batch_size


def _run_trial(
    self: "optuna.study.Study",
    func: "optuna.study.ObjectiveFuncType",
    catch: Tuple[Type[Exception], ...],
    gc_after_trial: bool,
) -> "optuna.trial.Trial":

    # Sync storage once at the beginning of the objective evaluation.
    self._storage.read_trials_from_remote_storage(self._study_id)

    trial_id = self._pop_waiting_trial_id()
    if trial_id is None:
        trial_id = self._storage.create_new_trial(self._study_id)
    trial = optuna.trial.Trial(self, trial_id)
    func(trial)
    return trial


@experimental("2.1.0")
def create_study(
    directions: List[str],
    study_name: Optional[str] = None,
    storage: Optional[Union[str, "optuna.storages.BaseStorage"]] = None,
    sampler: Optional["optuna.multi_objective.samplers.BaseMultiObjectiveSampler"] = None,
    load_if_exists: bool = False,
    batch_size: int = 1,
) -> BatchMultiObjectiveStudy:

    study = optuna.multi_objective.create_study(
        directions, study_name, storage, sampler, load_if_exists
    )
    return BatchMultiObjectiveStudy(study, batch_size)


@experimental("2.1.0")
def load_study(
    study_name: str,
    storage: Optional[Union[str, "optuna.storages.BaseStorage"]] = None,
    sampler: Optional["optuna.multi_objective.samplers.BaseMultiObjectiveSampler"] = None,
    batch_size: int = 1,
) -> BatchMultiObjectiveStudy:

    study = optuna.multi_objective.load_study(study_name, storage, sampler)
    return BatchMultiObjectiveStudy(study, batch_size)
