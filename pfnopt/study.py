import collections
import datetime
import multiprocessing
import multiprocessing.pool
import numpy as np
from six.moves import queue
import time
from typing import Any  # NOQA
from typing import Callable  # NOQA
from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA
from typing import Tuple  # NOQA
from typing import Type  # NOQA
from typing import Union  # NOQA

from pfnopt import logging
from pfnopt import pruners
from pfnopt import samplers
from pfnopt import storages
from pfnopt import structs
from pfnopt import trial as trial_module

ObjectiveFuncType = Callable[[trial_module.Trial], float]


class Study(object):

    """A study corresponds to an optimization task, i.e., a set of trials.

    This object provides interfaces to run a new trial, access trials' history, set/get
    user-defined attributes of the study itself.

    Args:
        study_uuid:
            Study's UUID. If this argument is set to None, a new study is created.
        storage:
            Storage object or its DB URL. If this argument is set to None, an InMemoryStorage is
            instantiated.
        sampler:
            Sampler object that implements background algorithm for value suggestion.
        pruner:
            Pruner object that decides early stopping of unpromising trials.

    """

    def __init__(
            self,
            study_uuid,  # type: str
            storage,  # type: Union[None, str, storages.BaseStorage]
            sampler=None,  # type: samplers.BaseSampler
            pruner=None,  # type: pruners.BasePruner
    ):
        # type: (...) -> None

        self.study_uuid = study_uuid
        self.storage = storages.get_storage(storage)
        self.sampler = sampler or samplers.TPESampler()
        self.pruner = pruner or pruners.MedianPruner()

        self.study_id = self.storage.get_study_id_from_uuid(study_uuid)
        self.logger = logging.get_logger(__name__)

    def __getstate__(self):
        # type: () -> Dict[Any, Any]
        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, state):
        # type: (Dict[Any, Any]) -> None
        self.__dict__.update(state)
        self.logger = logging.get_logger(__name__)

    @property
    def best_params(self):
        # type: () -> Dict[str, Any]

        return self.best_trial.params

    @property
    def best_value(self):
        # type: () -> float

        best_value = self.best_trial.value
        if best_value is None:
            raise ValueError('No trials are completed yet.')

        return best_value

    @property
    def best_trial(self):
        # type: () -> structs.FrozenTrial

        return self.storage.get_best_trial(self.study_id)

    @property
    def task(self):
        # type: () -> structs.StudyTask

        return self.storage.get_study_task(self.study_id)

    @property
    def trials(self):
        # type: () -> List[structs.FrozenTrial]

        return self.storage.get_all_trials(self.study_id)

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]

        return self.storage.get_study_user_attrs(self.study_id)

    def run(
            self,
            func,  # type: ObjectiveFuncType
            n_trials=None,  # type: Optional[int]
            timeout_seconds=None,  # type: Optional[float]
            n_jobs=1,  # type: int
            catch=(Exception,)  # type: Tuple[Type[Exception]]
    ):
        # type: (...) -> None

        if n_jobs == 1:
            self._run_sequential(func, n_trials, timeout_seconds, catch)
        else:
            self._run_parallel(func, n_trials, timeout_seconds, n_jobs, catch)

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None

        self.storage.set_study_user_attr(self.study_id, key, value)

    def _run_sequential(self, func, n_trials, timeout_seconds, catch):
        # type: (ObjectiveFuncType, Optional[int], Optional[float], Tuple[Type[Exception]]) -> None

        i_trial = 0
        time_start = datetime.datetime.now()
        while True:
            if n_trials is not None:
                if i_trial >= n_trials:
                    break
                i_trial += 1

            if timeout_seconds is not None:
                elapsed_seconds = (datetime.datetime.now() - time_start).total_seconds()
                if elapsed_seconds >= timeout_seconds:
                    break

            self._run_trial(func, catch)

    def _run_parallel(
            self,
            func,  # type: ObjectiveFuncType
            n_trials,  # type: Optional[int]
            timeout_seconds,  # type: Optional[float]
            n_jobs,  # type: int
            catch  # type: Tuple[Type[Exception]]
    ):
        # type: (...) -> None

        self.start_datetime = datetime.datetime.now()

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        if n_trials is not None:
            # The number of threads needs not to be larger than trials.
            n_jobs = min(n_jobs, n_trials)

            if n_trials == 0:
                return  # When n_jobs is zero, ThreadPool fails.

        pool = multiprocessing.pool.ThreadPool(n_jobs)  # type: ignore

        # A queue is passed to each thread. When True is received, then the thread continues
        # the evaluation. When False is received, then it quits.
        def func_child_thread(que):
            while que.get():
                self._run_trial(func, catch)
            self.storage.remove_session()

        que = multiprocessing.Queue(maxsize=n_jobs)  # type: ignore
        for _ in range(n_jobs):
            que.put(True)
        n_enqueued_trials = n_jobs
        imap_ite = pool.imap(func_child_thread, [que] * n_jobs, chunksize=1)

        while True:
            if timeout_seconds is not None:
                elapsed_timedelta = datetime.datetime.now() - self.start_datetime
                elapsed_seconds = elapsed_timedelta.total_seconds()
                if elapsed_seconds > timeout_seconds:
                    break

            if n_trials is not None:
                if n_enqueued_trials >= n_trials:
                    break

            try:
                que.put_nowait(True)
                n_enqueued_trials += 1
            except queue.Full:
                time.sleep(1)

        for _ in range(n_jobs):
            que.put(False)

        collections.deque(imap_ite, maxlen=0)  # Consume the iterator to wait for all threads.
        pool.terminate()
        que.close()
        que.join_thread()

    def _run_trial(self, func, catch):
        # type: (ObjectiveFuncType, Tuple[Type[Exception]]) -> trial_module.Trial

        trial_id = self.storage.create_new_trial_id(self.study_id)
        trial = trial_module.Trial(self, trial_id)

        try:
            result = func(trial)
        except catch as e:
            message = 'Setting trial status as {} because of the following error: {}'.format(
                structs.TrialState.FAIL, repr(e))
            self.logger.warning(message)
            self.storage.set_trial_state(trial_id, structs.TrialState.FAIL)
            self.storage.set_trial_system_attr(trial_id, 'fail_reason', message)
            return trial

        try:
            result = float(result)
        except (ValueError, TypeError,):
            message = 'Setting trial status as {} because the returned value from the ' \
                      'objective function cannot be casted to float. Returned value is: ' \
                      '{}'.format(structs.TrialState.FAIL, repr(result))
            self.logger.warning(message)
            self.storage.set_trial_state(trial_id, structs.TrialState.FAIL)
            self.storage.set_trial_system_attr(trial_id, 'fail_reason', message)
            return trial

        if np.isnan(result):
            message = 'Setting trial status as {} because the objective function returned ' \
                      '{}.'.format(structs.TrialState.FAIL, result)
            self.logger.warning(message)
            self.storage.set_trial_state(trial_id, structs.TrialState.FAIL)
            self.storage.set_trial_system_attr(trial_id, 'fail_reason', message)
            return trial

        trial.report(result)
        self.storage.set_trial_state(trial_id, structs.TrialState.COMPLETE)
        self._log_completed_trial(result)

        return trial

    def _log_completed_trial(self, value):
        # type: (float) -> None

        self.logger.info(
            'Finished a trial resulted in value: {}. '
            'Current best value is {} with parameters: {}.'.format(
                value, self.best_value, self.best_params))


def create_study(
        storage=None,  # type: Union[None, str, storages.BaseStorage]
        sampler=None,  # type: samplers.BaseSampler
        pruner=None,  # type: pruners.BasePruner
):
    # type: (...) -> Study

    """Create a new study.

    Args:
        storage:
            Storage object or its DB URL. If this argument is set to None, an InMemoryStorage is
            instantiated.
        sampler:
            Sampler object that implements background algorithm for value suggestion.
        pruner:
            Pruner object that decides early stopping of unpromising trials.

    Returns:
        A study object.

    """

    storage = storages.get_storage(storage)
    study_uuid = storage.get_study_uuid_from_id(storage.create_new_study_id())
    return Study(study_uuid=study_uuid, storage=storage, sampler=sampler, pruner=pruner)


def get_study(
        study,  # type: Union[str, Study]
        storage=None,  # type: Union[None, str, storages.BaseStorage]
        sampler=None,  # type: samplers.BaseSampler
        pruner=None,  # type: pruners.BasePruner
):
    # type: (...) -> Study

    """Return a given study object itself, or instantiate a study object with a given study UUID.

    Args:
        study:
            Study object or its UUID.
        storage:
            Storage object or its DB URL. If this argument is set to None, an InMemoryStorage is
            instantiated.
        sampler:
            Sampler object that implements background algorithm for value suggestion.
        pruner:
            Pruner object that decides early stopping of unpromising trials.

    Returns:
        A study object.

    """

    if isinstance(study, Study):
        if storage is not None:
            raise ValueError(
                'Do not give both study and storage objects at the same time. '
                'When a study is given, its associated storage will be used.')
        if sampler is not None:
            raise ValueError(
                'Do not give both study and sampler objects at the same time. '
                'When a study is given, its associated sampler will be used.')
        if pruner is not None:
            raise ValueError(
                'Do not give both study and pruner objects at the same time. '
                'When a study is given, its associated pruner will be used.')

        return study
    else:
        # `study` is expected to be a string and interpreted as a study UUID
        return Study(study_uuid=study, storage=storage, sampler=sampler, pruner=pruner)


def minimize(
        func,  # type: ObjectiveFuncType
        n_trials=None,  # type: Optional[int]
        timeout=None,  # type: Optional[float]
        n_jobs=1,  # type: int
        storage=None,  # type: Union[None, str, storages.BaseStorage]
        sampler=None,  # type: samplers.BaseSampler
        pruner=None,  # type: pruners.BasePruner
        study=None,  # type: Union[None, str, Study]
        catch=(Exception,)  # Tuple[Type[Exception]]
):
    # type: (...) -> Study

    """Minimize an objective function.

    Args:
        func:
            A callable that implements objective function.
        n_trials:
            The number of trials. If this argument is set to None, as many trials run as possible.
        timeout:
            Stop study after the given number of second(s). If this argument is set to None, as
            many trials run as possible.
        n_jobs:
            The number of parallel jobs. If this argument is set to -1, the number is set to CPU
            counts.
        storage:
            Storage object or its DB URL. If this argument is set to None, an InMemoryStorage is
            instantiated.
        sampler:
            Sampler object that implements background algorithm for value suggestion.
        pruner:
            Pruner object that decides early stopping of unpromising trials.
        study:
            Study object or its UUID. If this argument is set to None, a new study is created.
        catch:
            A study continues to run even when a trial raises one of exceptions specified in this
            argument. Default is (Exception,), where all non-exit exceptions are handled by this
            logic.

    Returns:
        A study object.

    """

    if study is not None:
        study = get_study(study, storage, sampler, pruner)
    else:
        if storage is not None:
            raise ValueError(
                'When specifying storage, please also specify a study UUID to continue a study. '
                'If you want to start a new study, please make a new one using create_study.')

        # We start a new study with a new in-memory storage.
        study = create_study(sampler=sampler, pruner=pruner)

    # Set up StudyTask as MINIMIZE.
    if study.task == structs.StudyTask.MAXIMIZE:
        raise ValueError(
            'Cannot run minimize task with study UUID {} because it already has been set up as a '
            'maximize task.'.format(study.study_uuid))
    study.storage.set_study_task(study.study_id, structs.StudyTask.MINIMIZE)

    study.run(func, n_trials, timeout, n_jobs, catch)
    return study


# TODO(akiba): implement me
def maximize():
    raise NotImplementedError


def get_all_study_summaries(storage):
    # type: (Union[str, storages.BaseStorage]) -> List[structs.StudySummary]

    """Get all history of studies stored in a specified storage.

    Args:
        storage:
            Storage object or its DB URL.

    Returns:
        List of study history summarized as StudySummary objects.

    """

    storage = storages.get_storage(storage)
    return storage.get_all_study_summaries()
