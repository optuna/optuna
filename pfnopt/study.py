import datetime
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


# TODO(Akiba): funcをStudyが持つ必要はないか？
class Study(object):

    def __init__(self, storage=None, sampler=None, pruner=None, study_id=0):
        # type: (storage_module.BaseStorage, samplers.BaseSampler, pruners.BasePruner, int) -> None
        self.study_id = study_id
        self.storage = storage or storage_module.InMemoryStorage()
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


# TODO(Akiba): Studyのメンバ関数にしない？
def minimize(
        func,  # type: Callable[[client_module.BaseClient], float]
        n_trials=None,  # type: Optional[int]
        timeout_seconds=None,  # type: Optional[int]
        study=None  # type: Optional[Study]
):
    # type: (...) -> Study
    if study is None:
        study = Study()

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

        trial_id = study.storage.create_new_trial_id(study.study_id)
        client = client_module.LocalClient(study, trial_id)
        result = func(client)
        client.complete(result)

    return study
