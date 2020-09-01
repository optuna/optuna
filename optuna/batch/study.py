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
        except optuna.exceptions.TrialPruned as e:
            for trial in trials:
                trial_id = trial._trial_id
                message = "Trial {} pruned. {}".format(trial.number, str(e))
                _logger.info(message)

                # Register the last intermediate value if present as the value of the trial.
                # TODO(hvy): Whether a pruned trials should have an actual value can be
                # discussed.
                frozen_trial = self._study._storage.get_trial(trial_id)
                last_step = frozen_trial.last_step
                if last_step is not None:
                    self._study._storage.set_trial_value(
                        trial_id, frozen_trial.intermediate_values[last_step]
                    )
                self._study._storage.set_trial_state(trial_id, optuna.trial.TrialState.PRUNED)
            raise
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

    @property
    def user_attrs(self) -> Dict[str, Any]:
        """Return user attributes.

        Returns:
            A dictionary containing all user attributes.
        """

        return self._study.user_attrs

    @property
    def system_attrs(self) -> Dict[str, Any]:
        """Return system attributes.

        Returns:
            A dictionary containing all system attributes.
        """

        return self._study.system_attrs

    def set_user_attr(self, key: str, value: Any) -> None:
        """Set a user attribute to the study.

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.
        """

        self._study.set_user_attr(key, value)

    def set_system_attr(self, key: str, value: Any) -> None:
        """Set a system attribute to the study.

        Note that Optuna internally uses this method to save system messages. Please use
        :func:`~optuna.multi_objective.study.MultiObjectiveStudy.set_user_attr`
        to set users' attributes.

        Args:
            key: A key string of the attribute.
            value: A value of the attribute. The value should be JSON serializable.

        """

        self._study.set_system_attr(key, value)

    def enqueue_trial(self, params: Dict[str, Any]) -> None:
        """Enqueue a trial with given parameter values.

        You can fix the next sampling parameters which will be evaluated in your
        objective function.

        Please refer to the documentation of :func:`optuna.study.Study.enqueue_trial`
        for further details.

        Args:
            params:
                Parameter values to pass your objective function.
        """

        self._study.enqueue_trial(params)

    @property
    def trials(self) -> List["optuna.trial.FrozenTrial"]:
        """Return all trials in the study.

        The returned trials are ordered by trial number.

        This is a short form of ``self.get_trials(deepcopy=True)``.

        Returns:
            A list of :class:`~optuna.FrozenTrial` objects.
        """

        return self.get_trials(deepcopy=True)

    def get_trials(self, deepcopy: bool = True) -> List["optuna.trial.FrozenTrial"]:
        """Return all trials in the study.

        The returned trials are ordered by trial number.

        For library users, it's recommended to use more handy
        :attr:`~optuna.study.Study.trials` property to get the trials instead.

        Args:
            deepcopy:
                Flag to control whether to apply ``copy.deepcopy()`` to the trials.
                Note that if you set the flag to :obj:`False`, you shouldn't mutate
                any fields of the returned trial. Otherwise the internal state of
                the study may corrupt and unexpected behavior may happen.

        Returns:
            A list of :class:`~optuna.FrozenTrial` objects.
        """

        return self._study.get_trials(deepcopy)

    @property
    def best_params(self):
        # type: () -> Dict[str, Any]
        """Return parameters of the best trial in the study.

        Returns:
            A dictionary containing parameters of the best trial.
        """

        return self._study.best_params

    @property
    def best_value(self) -> float:
        """Return the best objective value in the study.

        Returns:
            A float representing the best objective value.
        """

        return self._study.best_value

    @property
    def best_trial(self) -> "optuna.trial.FrozenTrial":
        """Return the best trial in the study.

        Returns:
            A :class:`~optuna.FrozenTrial` object of the best trial.
        """

        return self._study.best_trial

    @property
    def direction(self) -> "optuna.study.StudyDirection":
        """Return the direction of the study.

        Returns:
            A :class:`~optuna.study.StudyDirection` object.
        """

        return self._study.direction

    @property
    def _storage(self) -> "optuna.storages.BaseStorage":
        return self._study._storage

    @property
    def _study_id(self) -> int:
        return self._study._study_id


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
