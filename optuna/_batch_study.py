import copy
import gc
import math
from typing import Callable
from typing import List  # NOQA
from typing import Optional  # NOQA
from typing import Sequence  # NOQA
from typing import Tuple  # NOQA
from typing import Type  # NOQA

import optuna
import optuna.trial._batch
from optuna import trial as trial_module
from optuna import exceptions
from optuna import logging
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial

    ObjectiveFuncType = Callable[[trial_module.Trial], float]

BatchObjectiveFuncType = Callable[["optuna.trial._batch.BatchTrial"], Sequence[float]]

_logger = logging.get_logger(__name__)


class BatchStudy(optuna.study.Study):
    def __init__(self, study: "optuna.study.Study", batch_size: int) -> None:
        self._study = study
        super(BatchStudy, self).__init__(
            study_name=study.study_name,
            storage=study._storage,
            sampler=study.sampler,
            pruner=study.pruner,
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
        for trial in trials:
            frozen_trial = copy.deepcopy(self._storage.get_trial(trial._trial_id))
            for callback in self._callbacks:
                callback(self._study, frozen_trial)

    def _run_batch_trial(self,) -> List[trial_module.Trial]:

        # Sync storage once at the beginning of the objective evaluation.
        self._storage.read_trials_from_remote_storage(self._study_id)

        trials = []
        for _ in range(self.batch_size):
            trial_id = self._pop_waiting_trial_id()
            if trial_id is None:
                trial_id = self._storage.create_new_trial(self._study_id)
            trial = trial_module.Trial(self._study, trial_id)
            trial_number = trial.number
            trials.append(trial)
        batch_trial = optuna.trial._batch.BatchTrial(trials)

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
