import math
import types
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np

import optuna
from optuna._experimental import experimental
import optuna.logging

ObjectiveFuncType = Callable[["optuna.batch.trial.BatchTrial"], np.ndarray]
CallbackFuncType = Callable[["optuna.study.Study", "optuna.trial.FrozenTrial"], None]

_logger = optuna.logging.get_logger(__name__)


class _ObjectiveCallbackWrapper(object):
    def __init__(self, study: "optuna.study.Study", objective: ObjectiveFuncType, batch_size: int):
        self._study = study
        self._batch_size = batch_size
        self._objective = objective
        self._members = {}  # type: Dict[int, List[int]]

    def batch_objective(self, trial: "optuna.trial.Trial") -> float:
        trials = [trial]
        # Assume storage has already been synchronized.
        self._members[trial._trial_id] = []
        for _ in range(self._batch_size - 1):
            trial_id = self._study._pop_waiting_trial_id()
            if trial_id is None:
                trial_id = self._study._storage.create_new_trial(self._study._study_id)
            self._members[trial._trial_id].append(trial_id)
            trials.append(optuna.trial.Trial(self._study, trial_id))
        batch_trial = optuna.batch.trial.BatchTrial(trials)
        try:
            results = self._objective(batch_trial)
        except Exception as e:
            for trial in trials:
                message = "Trial {} failed because of the following error: {}".format(
                    trial.number, repr(e)
                )
                _logger.warning(message, exc_info=True)
                trial_id = trial._trial_id
                self._study._storage.set_trial_system_attr(trial_id, "fail_reason", message)
                self._study._storage.set_trial_state(trial_id, optuna.trial.TrialState.FAIL)
            raise

        for trial, result in zip(trials, results):
            trial_id = trial._trial_id
            trial_number = trial.number
            try:
                result = float(result)
            except (
                ValueError,
                TypeError,
            ):
                message = (
                    "Trial {} failed, because the returned value from the "
                    "objective function cannot be cast to float. Returned value is: "
                    "{}".format(trial_number, repr(result))
                )
                _logger.warning(message)
                self._study._storage.set_trial_system_attr(trial_id, "fail_reason", message)
                self._study._storage.set_trial_state(trial_id, optuna.trial.TrialState.FAIL)
                continue

            if math.isnan(result):
                message = "Trial {} failed because the objective function returned {}.".format(
                    trial_number, result
                )
                _logger.warning(message)
                self._study._storage.set_trial_system_attr(trial_id, "fail_reason", message)
                self._study._storage.set_trial_state(trial_id, optuna.trial.TrialState.FAIL)
                continue

            self._study._storage.set_trial_value(trial_id, result)
            self._study._storage.set_trial_state(trial_id, optuna.trial.TrialState.COMPLETE)
            self._study._log_completed_trial(trial, result)
        return results[0]

    def wrap_callback(self, callback: CallbackFuncType) -> CallbackFuncType:
        def _callback(study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> None:
            callback(study, trial)
            for member_id in self._members[trial._trial_id]:
                _trial = study._storage.get_trial(member_id)
                callback(study, _trial)

        return _callback


class BatchStudy(object):
    def __init__(self, study: "optuna.study.Study", batch_size: int):
        self._study = study
        self._batch_size = batch_size

    def __getattr__(self, attr_name: str) -> Any:
        return getattr(self._study, attr_name)

    @property
    def batch_size(self) -> int:
        """Return the size of batches.

        Returns:
            Size of batches.
        """

        return self._batch_size

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

        try:
            self._study._org_run_trial = self._study._run_trial  # type: ignore
            self._study._run_trial = types.MethodType(_run_trial, self._study)  # type: ignore
            self._study.optimize(
                wrapper.batch_objective,
                timeout=timeout,
                n_trials=n_trials,
                catch=catch,
                callbacks=wrapped_callbacks,
                gc_after_trial=gc_after_trial,
                show_progress_bar=show_progress_bar,
            )
        finally:
            self._study._run_trial = self._study._org_run_trial  # type: ignore
            pass


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
    storage: Optional[Union[str, "optuna.storages.BaseStorage"]] = None,
    sampler: Optional["optuna.samplers.BaseSampler"] = None,
    pruner: Optional["optuna.pruners.BasePruner"] = None,
    study_name: Optional[str] = None,
    direction: str = "minimize",
    load_if_exists: bool = False,
    batch_size: int = 1,
) -> BatchStudy:

    study = optuna.create_study(storage, sampler, pruner, study_name, direction, load_if_exists)
    return BatchStudy(study, batch_size)


@experimental("2.1.0")
def load_study(
    study_name: str,
    storage: Union[str, "optuna.storages.BaseStorage"],
    sampler: Optional["optuna.samplers.BaseSampler"] = None,
    pruner: Optional["optuna.pruners.BasePruner"] = None,
    batch_size: int = 1,
) -> BatchStudy:

    study = optuna.load_study(study_name, storage, sampler, pruner)
    return BatchStudy(study, batch_size)
