import datetime
import multiprocessing
import multiprocessing.pool
from typing import Any  # NOQA
from typing import Callable  # NOQA
from typing import Dict  # NOQA
from typing import Iterable  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA
from typing import Union  # NOQA

from pfnopt import client as client_module
from pfnopt import logging
from pfnopt import pruners
from pfnopt import samplers
from pfnopt import storages
from pfnopt import trial  # NOQA

ObjectiveFuncType = Callable[[client_module.BaseClient], float]


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
        # type: () -> trial.Trial

        return self.storage.get_best_trial(self.study_id)

    @property
    def trials(self):
        # type: () -> List[trial.Trial]

        return self.storage.get_all_trials(self.study_id)

    def run(self, func, n_trials=None, timeout_seconds=None, n_jobs=1):
        # type: (ObjectiveFuncType, Optional[int], Optional[float], int) -> None

        if n_jobs == 1:
            self._run_sequential(func, n_trials, timeout_seconds)
        else:
            self._run_parallel(func, n_trials, timeout_seconds, n_jobs)

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
            client = client_module.LocalClient(self, trial_id)
            result = func(client)
            client.complete(result)
            self._log_completed_trial(trial_id, result)

    def _run_parallel(self, func, n_trials, timeout_seconds, n_jobs):
        # type: (ObjectiveFuncType, Optional[int], Optional[float], int) -> None

        if isinstance(self.storage, storages.RDBStorage):
            raise TypeError('Parallel run with RDBStorage is not supported.')

        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()

        pool = multiprocessing.pool.ThreadPool(n_jobs)  # type: ignore

        def f(_):
            trial_id = self.storage.create_new_trial_id(self.study_id)
            client = client_module.LocalClient(self, trial_id)
            result = func(client)
            client.complete(result)
            self._log_completed_trial(trial_id, result)

        self.start_datetime = datetime.datetime.now()

        if n_trials is not None:
            ite = range(n_trials)  # type: Iterable[int]
        else:
            ite = iter(int, 1)  # Infinite iterator

        imap_ite = pool.imap(f, ite, chunksize=1)
        while True:
            if timeout_seconds is None:
                to = None
            else:
                elapsed_timedelta = datetime.datetime.now() - self.start_datetime
                elapsed_seconds = elapsed_timedelta.total_seconds()
                to = (timeout_seconds - elapsed_seconds)

            try:
                imap_ite.next(timeout=to)  # type: ignore
            except (StopIteration, multiprocessing.TimeoutError):  # type: ignore
                break

        pool.terminate()

    def _log_completed_trial(self, trial_id, value):
        self.logger.info(
            'Finished trial {} resulted in value: {}. '
            'Current best value is {} with parameters: {}.'.format(
                trial_id, value, self.best_value, self.best_params))


def create_study(
        storage=None,  # type: Union[None, str, storages.BaseStorage]
        sampler=None,  # type: samplers.BaseSampler
        pruner=None,  # type: pruners.BasePruner
):
    # type: (...) -> Study

    storage = storages.get_storage(storage)
    study_uuid = storage.get_study_uuid_from_id(storage.create_new_study_id())
    return Study(study_uuid=study_uuid, storage=storage, sampler=sampler, pruner=pruner)


def minimize(
        func,  # type: ObjectiveFuncType
        n_trials=None,  # type: Optional[int]
        timeout_seconds=None,  # type: Optional[float]
        n_jobs=1,  # type: int
        storage=None,  # type: Union[None, str, storages.BaseStorage]
        sampler=None,  # type: samplers.BaseSampler
        pruner=None,  # type: pruners.BasePruner
        study_uuid=None,  # type: str
        study=None,  # type: Study
):
    # type: (...) -> Study

    if study is not None:
        # We continue the given study
        if storage is not None:
            raise ValueError(
                'Do not specify both study and storage at the same time. '
                'When a study is given, its associated storage will be used.')
        if sampler is not None:
            raise ValueError(
                'Do not specify both study and sampler at the same time. '
                'When a study is given, its associated sampler will be used.')
        if pruner is not None:
            raise ValueError(
                'Do not specify both study and pruner at the same time. '
                'When a study is given, its associated pruner will be used.')
    elif storage is not None:
        # We connect to an existing study in the storage
        if study_uuid is None:
            raise ValueError(
                'When specifying storage, please also specify study_uuid to continue a study. '
                'If you want to start a new study, please make a new one using create_study.')
        storage = storages.get_storage(storage)
        study = Study(study_uuid, storage, sampler, pruner)
    else:
        # We start a new study with a new in-memory storage
        study = create_study(sampler=sampler, pruner=pruner)

    study.run(func, n_trials, timeout_seconds, n_jobs)
    return study


# TODO(akiba): implement me
def maximize():
    raise NotImplementedError
