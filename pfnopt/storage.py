import copy

from . import trial


class InMemoryStorage(object):

    def __init__(self):
        self.trials = []

    def get_param(self, study_id, trial_id, param_name):
        raise NotImplementedError

    def report_param(self, study_id, trial_id, param_name, value):
        assert study_id == 0  # TODO
        self.trials[trial_id].params[param_name] = value

    def report_result(self, study_id, trial_id, result):
        assert study_id == 0  # TODO
        self.trials[trial_id].result = result

    def create_new_trial_id(self, study_id):
        assert study_id == 0
        trial_id = len(self.trials)
        self.trials.append(trial.Trial(trial_id, {}, None))
        return trial_id

    def collect_param_result_pairs(self, study_id, param_name):
        assert study_id == 0

        return [
            (t.params[param_name], t.result)
            for t in self.trials
            if param_name in t.params and t.result is not None
        ]

    def get_best_trial(self):
        # TODO: non-empty check

        best_trial = min(
            (t for t in self.trials if t.result is not None),
            key=lambda t: t.result)

        return copy.deepcopy(best_trial)

    def get_all_trials(self):
        return copy.deepcopy(self.trials)
