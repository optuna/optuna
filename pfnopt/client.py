class BaseClient(object):

    def ask_uniform(self, name, low, high):
        return self._ask(name, {'kind': 'uniform', 'low': low, 'high': high})

    def report_result(self, result):
        raise NotImplementedError

    def _ask(self, name, distribution):
        raise NotImplementedError


class LocalClient(BaseClient):

    """Client that communicates with local study object"""

    def __init__(self, study, trial_id):
        self.study = study
        self.trial_id = trial_id

    def _ask(self, name, distribution):
        # TODO: if already sampled, return the recorded value
        # TODO: check that distribution is the same

        pairs = self.study.storage.collect_param_result_pairs(
            self.study.study_id, name)
        val = self.study.sampler.sample(distribution, pairs)
        self.study.storage.report_param(
            self.study.study_id, self.trial_id, name, val
        )
        return val

    def report_result(self, result):
        self.study.storage.report_result(
            self.study.study_id, self.trial_id, result)
