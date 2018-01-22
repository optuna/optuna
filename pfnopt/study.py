from . import storage as storage_module
from . import samplers
from . import client as client_module


# TODO: 実験継続と新規実験のどっちも簡単にできるインターフェースを考える必要あり


class Study(object):

    def __init__(self, storage=None, sampler=None, study_id=0):
        self.study_id = study_id
        self.storage = storage or storage_module.InMemoryStorage()
        self.sampler = sampler or samplers.TPESampler()

    @property
    def best_params(self):
        return self.best_trial.params

    @property
    def best_result(self):
        return self.best_trial.result

    @property
    def best_trial(self):
        return self.storage.get_best_trial()

    @property
    def trials(self):
        return self.storage.get_all_trials()


def minimize(func, n_trials, study=None):
    if study is None:
        study = Study()

    for _ in range(n_trials):
        trial_id = study.storage.create_new_trial_id(study.study_id)
        client = client_module.LocalClient(study, trial_id)
        result = func(client)
        client.report_result(result)

    return study
