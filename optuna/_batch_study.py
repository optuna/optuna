import copy
import gc
import math
from typing import Callable
from typing import List  # NOQA
from typing import Optional  # NOQA
from typing import Sequence  # NOQA
from typing import Tuple  # NOQA
from typing import Type  # NOQA
from typing import Union  # NOQA

import numpy as np

import optuna
from optuna import exceptions
from optuna import logging
from optuna.multi_objective.study import MultiObjectiveStudy
from optuna.multi_objective.trial import FrozenMultiObjectiveTrial
from optuna import trial as trial_module
from optuna.trial._batch import BatchMultiObjectiveTrial
from optuna.trial._batch import BatchTrial
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial

    ObjectiveFuncType = Callable[[trial_module.Trial], float]

BatchObjectiveFuncType = Callable[[BatchTrial], np.ndarray]
BatchMultiObjectiveFuncType = Callable[[BatchMultiObjectiveTrial], Sequence[np.ndarray]]
MultiObjectiveCallbackFuncType = Callable[
    [
        "optuna.multi_objective.study.MultiObjectiveStudy",
        "optuna.multi_objective.trial.FrozenMultiObjectiveTrial",
    ],
    None,
]

_logger = logging.get_logger(__name__)


class BatchStudy(optuna.study.Study):
    def __init__(
        self, study: Union["optuna.study.Study", MultiObjectiveStudy], batch_size: int
    ) -> None:
        self._study = study

        if isinstance(study, MultiObjectiveStudy):
            study_name = study._study.study_name
            sampler = None
            pruner = None
        else:
            study_name = study.study_name
            sampler = study.sampler
            pruner = study.pruner

        super().__init__(
            study_name=study_name, storage=study._storage, sampler=sampler, pruner=pruner,
        )
        self.batch_size = batch_size

    def optimize(
        self,
        func,  # type: ObjectiveFuncType
        n_trials=None,  # type: Optional[int]
        timeout=None,  # type: Optional[float]
        n_jobs=1,  # type: int
        catch=(),  # type: Tuple[Type[Exception], ...]
        callbacks=None,  # type: Optional[List[Callable[[Study, FrozenTrial], None]]]
        gc_after_trial=False,  # type: bool
        show_progress_bar=False,  # type: bool
    ) -> None:
        raise NotImplementedError

    def batch_optimize(
        self,
        func: BatchObjectiveFuncType,
        n_batches: Optional[int] = None,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        catch: Tuple[Type[Exception], ...] = (),
        callbacks: Optional[List[Callable[["Study", "FrozenTrial"], None]]] = None,
        gc_after_trial: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        if not isinstance(catch, tuple):
            raise TypeError(
                "The catch argument is of type '{}' but must be a tuple.".format(
                    type(catch).__name__
                )
            )

        self._batch_func = func
        self._catch = catch
        self._callbacks = callbacks
        self._gc_after_trial = gc_after_trial

        self._optimize(n_batches, timeout, n_jobs, show_progress_bar)

    def _run_trial_and_callbacks(self,) -> None:

        trials = self._run_batch_trial()
        if self._callbacks is None:
            return

        if isinstance(self._study, MultiObjectiveStudy):
            study = self._study._study
        else:
            study = self._study
        for trial in trials:
            frozen_trial = copy.deepcopy(self._storage.get_trial(trial._trial_id))
            for callback in self._callbacks:
                callback(study, frozen_trial)

    def _run_batch_trial(self,) -> List[trial_module.Trial]:

        # Sync storage once at the beginning of the objective evaluation.
        self._storage.read_trials_from_remote_storage(self._study_id)

        trials = []
        for _ in range(self.batch_size):
            trial_id = self._pop_waiting_trial_id()
            if trial_id is None:
                trial_id = self._storage.create_new_trial(self._study_id)
            if isinstance(self._study, MultiObjectiveStudy):
                study = self._study._study
            else:
                study = self._study
            trial = trial_module.Trial(study, trial_id)
            trial_number = trial.number
            trials.append(trial)
        batch_trial = BatchTrial(trials)

        try:
            # Evaluate the batched objective function.
            results = self._batch_func(batch_trial)
        except exceptions.TrialPruned as e:
            message = "Trial {} to {} are pruned. {}".format(
                trials[0].number, trials[-1].number, str(e)
            )
            _logger.info(message)

            # Register the last intermediate value if present as the value of the trial.
            # TODO(hvy): Whether a pruned trials should have an actual value can be discussed.
            for trial in trials:
                trial_id = trial._trial_id
                frozen_trial = self._storage.get_trial(trial_id)
                last_step = frozen_trial.last_step
                if last_step is not None:
                    self._storage.set_trial_value(
                        trial_id, frozen_trial.intermediate_values[last_step]
                    )
                self._storage.set_trial_state(trial_id, trial_module.TrialState.PRUNED)
            return trials
        except Exception as e:
            message = "Trial {} to {} failed because of the following error: {}".format(
                trials[0].number, trials[-1].number, repr(e)
            )
            _logger.warning(message, exc_info=True)
            for trial in trials:
                trial_id = trial._trial_id
                self._storage.set_trial_system_attr(trial_id, "fail_reason", message)
                self._storage.set_trial_state(trial_id, trial_module.TrialState.FAIL)

            if isinstance(e, self._catch):
                return trials
            raise
        finally:
            # The following line mitigates memory problems that can be occurred in some
            # environments (e.g., services that use computing containers such as CircleCI).
            # Please refer to the following PR for further details:
            # https://github.com/optuna/optuna/pull/325.
            if self._gc_after_trial:
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
                self._storage.set_trial_state(trial_id, trial_module.TrialState.FAIL)
                continue

            if math.isnan(result):
                message = "Trial {} failed, because the objective function returned {}.".format(
                    trial_number, result
                )
                _logger.warning(message)
                self._storage.set_trial_system_attr(trial_id, "fail_reason", message)
                self._storage.set_trial_state(trial_id, trial_module.TrialState.FAIL)
                continue

            self._storage.set_trial_value(trial_id, result)
            self._storage.set_trial_state(trial_id, trial_module.TrialState.COMPLETE)
            self._log_completed_trial(trial, result)
        return trials


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
