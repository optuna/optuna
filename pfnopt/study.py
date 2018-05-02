import collections
import datetime
import multiprocessing
import multiprocessing.pool
from six.moves import queue
import time
from typing import Any  # NOQA
from typing import Callable  # NOQA
from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA
from typing import Union  # NOQA

from pfnopt import frozen_trial  # NOQA
from pfnopt import logging
from pfnopt import pruners
from pfnopt import samplers
from pfnopt import storages
from pfnopt import trial as trial_module

ObjectiveFuncType = Callable[[trial_module.Trial], float]


class Study(object):

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

        return self.best_trial.value

    @property
    def best_trial(self):
        # type: () -> frozen_trial.FrozenTrial

        return self.storage.get_best_trial(self.study_id)

    @property
    def trials(self):
        # type: () -> List[frozen_trial.FrozenTrial]

        return self.storage.get_all_trials(self.study_id)

    @property
    def user_attrs(self):
        # type: () -> Dict[str, Any]

        return self.storage.get_study_user_attrs(self.study_id)

    def run(self, func, n_trials=None, timeout_seconds=None, n_jobs=1):
        # type: (ObjectiveFuncType, Optional[int], Optional[float], int) -> None

        if n_jobs == 1:
            self._run_sequential(func, n_trials, timeout_seconds)
        else:
            self._run_parallel(func, n_trials, timeout_seconds, n_jobs)

    def set_user_attr(self, key, value):
        # type: (str, Any) -> None

        self.storage.set_study_user_attr(self.study_id, key, value)

    def _run_sequential(self, func, n_trials, timeout_seconds):
        # type: (ObjectiveFuncType, Optional[int], Optional[float]) -> None

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

            trial_id = self.storage.create_new_trial_id(self.study_id)
            trial = trial_module.Trial(self, trial_id)
            result = func(trial)
            trial.report(result)
            trial.complete()
            self._log_completed_trial(trial_id, result)

    def _run_parallel(self, func, n_trials, timeout_seconds, n_jobs):
        # type: (ObjectiveFuncType, Optional[int], Optional[float], int) -> None

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
                trial_id = self.storage.create_new_trial_id(self.study_id)
                trial = trial_module.Trial(self, trial_id)
                result = func(trial)
                trial.report(result)
                trial.complete()
                self._log_completed_trial(trial_id, result)
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

    def _log_completed_trial(self, trial_id, value):
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
):
    # type: (...) -> Study

    if study is not None:
        study = get_study(study, storage, sampler, pruner)
    else:
        if storage is not None:
            raise ValueError(
                'When specifying storage, please also specify a study UUID to continue a study. '
                'If you want to start a new study, please make a new one using create_study.')

        # We start a new study with a new in-memory storage.
        study = create_study(sampler=sampler, pruner=pruner)

    study.run(func, n_trials, timeout, n_jobs)
    return study


# TODO(akiba): implement me
def maximize():
    raise NotImplementedError
