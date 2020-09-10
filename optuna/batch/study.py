import copy
from functools import partial
import gc
import math
import types
from typing import Any
from typing import Callable
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

        n_trials = math.ceil(n_batches / n_jobs) if n_batches is not None else None

        self._study._run_trial_and_callbacks = types.MethodType(  # type: ignore
            partial(_run_trial_and_callbacks, batch_func=objective, batch_size=self._batch_size),
            self._study,
        )
        self._study.optimize(
            lambda _: 0,
            timeout=timeout,
            n_trials=n_trials,
            catch=catch,
            callbacks=callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar,
        )


def _run_trial_and_callbacks(
    self: "optuna.study.Study",
    func: "optuna.study.ObjectiveFuncType",
    catch: Tuple[Type[Exception], ...],
    callbacks: Optional[List[Callable[["optuna.study.Study", "optuna.trial.FrozenTrial"], None]]],
    gc_after_trial: bool,
    batch_func: ObjectiveFuncType,
    batch_size: int,
) -> None:

    trials = _run_trial(self, batch_func, catch, gc_after_trial, batch_size)
    if callbacks is None:
        return

    for trial in trials:
        frozen_trial = copy.deepcopy(self._storage.get_trial(trial._trial_id))
        for callback in callbacks:
            callback(self, frozen_trial)


def _run_trial(
    self: "optuna.study.Study",
    func: ObjectiveFuncType,
    catch: Tuple[Type[Exception], ...],
    gc_after_trial: bool,
    batch_size: int,
) -> List["optuna.trial.Trial"]:

    # Sync storage once at the beginning of the objective evaluation.
    self._storage.read_trials_from_remote_storage(self._study_id)

    trials = []
    for _ in range(batch_size):
        trial_id = self._pop_waiting_trial_id()
        if trial_id is None:
            trial_id = self._storage.create_new_trial(self._study_id)
        trial = optuna.trial.Trial(self, trial_id)
        trial_number = trial.number
        trials.append(trial)
    batch_trial = optuna.batch.trial.BatchTrial(trials)

    try:
        # Evaluate the batched objective function.
        results = func(batch_trial)
    except Exception as e:
        message = "Trial {} to {} failed because of the following error: {}".format(
            trials[0].number, trials[-1].number, repr(e)
        )
        _logger.warning(message, exc_info=True)
        for trial in trials:
            trial_id = trial._trial_id
            self._storage.set_trial_system_attr(trial_id, "fail_reason", message)
            self._storage.set_trial_state(trial_id, optuna.trial.TrialState.FAIL)

        if isinstance(e, catch):
            return trials
        raise
    finally:
        # The following line mitigates memory problems that can be occurred in some
        # environments (e.g., services that use computing containers such as CircleCI).
        # Please refer to the following PR for further details:
        # https://github.com/optuna/optuna/pull/325.
        if gc_after_trial:
            gc.collect()

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
            self._storage.set_trial_system_attr(trial_id, "fail_reason", message)
            self._storage.set_trial_state(trial_id, optuna.trial.TrialState.FAIL)
            continue

        if math.isnan(result):
            message = "Trial {} failed, because the objective function returned {}.".format(
                trial_number, result
            )
            _logger.warning(message)
            self._storage.set_trial_system_attr(trial_id, "fail_reason", message)
            self._storage.set_trial_state(trial_id, optuna.trial.TrialState.FAIL)
            continue

        self._storage.set_trial_value(trial_id, result)
        self._storage.set_trial_state(trial_id, optuna.trial.TrialState.COMPLETE)
        self._log_completed_trial(trial, result)
    return trials


@experimental("2.1.0")
def create_study(
    storage: Optional[Union[str, "optuna.storages.BaseStorage"]] = None,
    sampler: Optional["optuna.samplers.BaseSampler"] = None,
    study_name: Optional[str] = None,
    direction: str = "minimize",
    load_if_exists: bool = False,
    batch_size: int = 1,
) -> BatchStudy:

    study = optuna.create_study(storage, sampler, None, study_name, direction, load_if_exists)
    return BatchStudy(study, batch_size)


@experimental("2.1.0")
def load_study(
    study_name: str,
    storage: Union[str, "optuna.storages.BaseStorage"],
    sampler: Optional["optuna.samplers.BaseSampler"] = None,
    batch_size: int = 1,
) -> BatchStudy:

    study = optuna.load_study(study_name, storage, sampler, None)
    return BatchStudy(study, batch_size)
