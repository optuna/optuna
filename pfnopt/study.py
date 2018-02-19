import datetime
import multiprocessing
import multiprocessing.pool
import os
from typing import Any  # NOQA
from typing import Callable  # NOQA
from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA

from pfnopt import client as client_module
from pfnopt import pruners
from pfnopt import samplers
from pfnopt import storage as storage_module
from pfnopt import trial  # NOQA


# TODO(Akiba): 実験継続と新規実験のどっちも簡単にできるインターフェースを考える必要あり


class Study(object):

    def __init__(self, storage=None, sampler=None, pruner=None, study_id=0):
        # type: (storage_module.BaseStorage, samplers.BaseSampler, pruners.BasePruner, int) -> None
        self.study_id = study_id
        self.storage = storage
        self.sampler = sampler or samplers.TPESampler()
        self.pruner = pruner or pruners.MedianPruner()

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

    def run(self, func, n_trials, timeout_seconds=None,
            n_jobs=1, parallelism_backend='process'):
        if n_jobs == 1:
            return self._run_sequential(func, n_trials, timeout_seconds)
        else:
            return self._run_parallel(func, n_trials, timeout_seconds, n_jobs, parallelism_backend)

    def _run_sequential(self, func, n_trials, timeout_seconds):
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

    def _run_parallel(self, func, n_trials, timeout_seconds, n_jobs, parallelism_backend):
        if parallelism_backend == 'thread':
            pool_class = multiprocessing.pool.ThreadPool
        # TODO: doesn't work with any configuration
        # elif parallelism_backend == 'process':
        #    pool_class = multiprocessing.Pool
        else:
            raise ValueError('Unknown parallelism backend specified: {}'.format(parallelism_backend))

        if n_jobs == -1:
            n_jobs = os.cpu_count()

        pool = pool_class(n_jobs)

        def f(_):
            trial_id = self.storage.create_new_trial_id(self.study_id)
            client = client_module.LocalClient(self, trial_id)
            result = func(client)
            client.complete(result)

        pool.map(f, range(n_trials), chunksize=1)  # TODO: timeout


# TODO: add some study-wise configuration (e.g., minimize? maximize?)
def create_new_study(storage):
    study_id = storage.create_new_study_id()
    return Study(study_id=study_id, storage=storage)


# TODO: Studyのメンバ関数にしない？
def minimize(
        func,  # type: Callable[[client_module.BaseClient], float]
        n_trials=None,  # type: Optional[int]
        timeout_seconds=None,  # type: Optional[int]
        n_jobs=1,  # type: int
        parallelism_backend='thread'  # type: str
):
    # type: (...) -> Study

    storage = storage_module.InMemoryStorage()
    study = create_new_study(storage)
    study.run(func, n_trials, timeout_seconds, n_jobs, parallelism_backend)
    return study


# TODO: implement me
def maximize():
    raise NotImplementedError
