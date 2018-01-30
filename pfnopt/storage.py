import copy
import numpy as np

from . import trial


class InMemoryStorage(object):

    def __init__(self):
        self.trials = []

    def create_new_trial_id(self, study_id):
        assert study_id == 0
        trial_id = len(self.trials)
        self.trials.append(trial.Trial(trial_id, {}, None))
        return trial_id

    def get_param(self, study_id, trial_id, param_name):
        raise NotImplementedError

    def set_param(self, study_id, trial_id, param_name, value):
        assert study_id == 0  # TODO
        self.trials[trial_id].params[param_name] = value

    def set_result(self, study_id, trial_id, result):
        assert study_id == 0  # TODO
        self.trials[trial_id].result = result

    def set_intermediate_result(self, study_id, trial_id, step, intermediate_result):
        assert study_id == 0  # TODO
        self.trials[trial_id].intermediate_results[step] = intermediate_result

    #
    # Methods for result analysis
    #

    def get_best_trial(self):
        # TODO: non-empty check

        best_trial = min(
            (t for t in self.trials if t.result is not None),
            key=lambda t: t.result)

        return copy.deepcopy(best_trial)

    def get_all_trials(self):
        return copy.deepcopy(self.trials)

    #
    # Methods for the TPE sampler
    #

    def get_param_result_pairs(self, study_id, param_name):
        assert study_id == 0

        return [
            (t.params[param_name], t.result)
            for t in self.trials
            if param_name in t.params and t.result is not None
        ]

    #
    # Methods for the median pruner
    #

    def get_best_intermediate_result_over_steps(self, study_id, trial_id):
        assert study_id == 0
        return min(self.trials[trial_id].intermediate_results.values())

    def get_median_intermediate_result_over_trials(self, study_id, step):
        assert study_id == 0

        return np.median([
            t.intermediate_results[step]
            for t in self.trials
            if step in t.intermediate_results
        ])
