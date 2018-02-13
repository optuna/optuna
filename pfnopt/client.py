import datetime

from . import trial


# TODO: don't we need distribution class?

class BaseClient(object):

    def sample_uniform(self, name, low, high):
        return self._sample(name, {'kind': 'uniform', 'low': low, 'high': high})

    def sample_loguniform(self, name, low, high):
        return self._sample(name, {'kind': 'loguniform', 'low': low, 'high': high})

    def complete(self, result):
        raise NotImplementedError

    def prune(self, step, current_result):
        raise NotImplementedError

    @property
    def params(self):
        raise NotImplementedError

    @property
    def info(self):
        raise NotImplementedError

    def _sample(self, name, distribution):
        raise NotImplementedError


class LocalClient(BaseClient):

    """Client that communicates with local study object"""

    def __init__(self, study, trial_id):
        self.study = study
        self.trial_id = trial_id

        self.study_id = self.study.study_id
        self.storage = self.study.storage

        self.storage.set_trial_system_attr(
            self.study_id, self.trial_id,
            'datetime_start', datetime.datetime.now())

    def _sample(self, name, distribution):
        # TODO: if already sampled, return the recorded value
        # TODO: check that distribution is the same

        pairs = self.storage.get_trial_param_result_pairs(
            self.study_id, name)
        val = self.study.sampler.sample(distribution, pairs)
        self.storage.set_trial_param(
            self.study_id, self.trial_id, name, val)
        return val

    def complete(self, result):
        self.storage.set_trial_value(
            self.study_id, self.trial_id, result)
        self.storage.set_trial_state(
            self.study_id, self.trial_id, trial.State.COMPLETE)
        self.storage.set_trial_system_attr(
            self.study_id, self.trial_id,
            'datetime_complete', datetime.datetime.now())

    def prune(self, step, current_result):
        self.storage.set_trial_intermediate_value(
            self.study_id, self.trial_id, step, current_result)
        ret = self.study.pruner.prune(
            self.storage, self.study_id, self.trial_id, step)
        return ret

    @property
    def params(self):
        return self.storage.get_trial_params(
            self.study_id, self.trial_id)

    @property
    def info(self):
        return self.storage.get_trial_system_attrs(
            self.study_id, self.trial_id)
