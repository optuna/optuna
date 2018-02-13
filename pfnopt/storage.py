import copy
import numpy as np

from . import trial


class InMemoryStorage(object):

    def __init__(self):
        self.trials = []

    def create_new_trial_id(self, study_id):
        assert study_id == 0
        trial_id = len(self.trials)
        self.trials.append(trial.Trial(trial_id))
        return trial_id

    def set_state(self, study_id, trial_id, state):
        assert study_id == 0
        self.trials[trial_id].state = state

    def get_param_dict(self, study_id, trial_id):
        assert study_id == 0
        return copy.deepcopy(self.trials[trial_id].params)

    def get_param(self, study_id, trial_id, param_name):
        raise NotImplementedError

    def set_param(self, study_id, trial_id, param_name, param_value):
        assert study_id == 0  # TODO
        self.trials[trial_id].params[param_name] = param_value

    def set_value(self, study_id, trial_id, value):
        assert study_id == 0  # TODO
        self.trials[trial_id].value = value

    def set_intermediate_value(self, study_id, trial_id, step, intermediate_value):
        assert study_id == 0  # TODO
        self.trials[trial_id].intermediate_values[step] = intermediate_value

    def get_system_attrs(self, study_id, trial_id):
        assert study_id == 0
        return copy.deepcopy(self.trials[trial_id].system_attrs)

    def set_system_attr(self, study_id, trial_id, attr_name, attr_value):
        assert study_id == 0
        self.trials[trial_id].system_attrs[attr_name] = attr_value


    #
    # Methods for result analysis
    #

    def get_best_trial(self):
        # TODO: non-empty check

        best_trial = min(
            (t for t in self.trials if t.state is trial.State.COMPLETE),
            key=lambda t: t.value)

        return copy.deepcopy(best_trial)

    def get_all_trials(self):
        return copy.deepcopy(self.trials)

    #
    # Methods for the TPE sampler
    #

    def get_param_result_pairs(self, study_id, param_name):
        assert study_id == 0

        return [
            (t.params[param_name], t.value)
            for t in self.trials
            if param_name in t.params and t.value is trial.State.COMPLETE
            # TODO: We also want to use pruned results
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
