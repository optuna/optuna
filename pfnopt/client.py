class BaseClient(object):

    def sample_uniform(self, name, low, high):
        return self._sample(name, {'kind': 'uniform', 'low': low, 'high': high})

    def complete(self, result):
        raise NotImplementedError

    def prune(self, step, current_result):
        pass

    def _sample(self, name, distribution):
        raise NotImplementedError


class LocalClient(BaseClient):

    """Client that communicates with local study object"""

    def __init__(self, study, trial_id):
        self.study = study
        self.trial_id = trial_id

    def _sample(self, name, distribution):
        # TODO: if already sampled, return the recorded value
        # TODO: check that distribution is the same

        pairs = self.study.storage.collect_param_result_pairs(
            self.study.study_id, name)
        val = self.study.sampler.sample(distribution, pairs)
        self.study.storage.report_param(
            self.study.study_id, self.trial_id, name, val)
        return val

    def complete(self, result):
        self.study.storage.report_result(
            self.study.study_id, self.trial_id, result)

    def prune(self, step, current_result):
        self.study.storage.report_intermediate_result(
            self.study.study_id, self.trial_id, step, current_result)
        ret = self.study.pruner.prune(
            self.study.storage, self.study.study_id, self.trial_id, step)
        return ret
