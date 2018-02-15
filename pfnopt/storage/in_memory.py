import copy

from pfnopt import trial
from . import _base


class InMemoryStorage(_base.BaseStorage):

    def __init__(self):
        self.study_attrs = {}
        self.trials = []

    def create_new_trial_id(self, study_id):
        assert study_id == 0  # TODO
        trial_id = len(self.trials)
        self.trials.append(trial.Trial(trial_id))
        return trial_id

    def set_trial_state(self, study_id, trial_id, state):
        assert study_id == 0  # TODO
        self.trials[trial_id].state = state

    def set_trial_param(self, study_id, trial_id, param_name, param_value):
        assert study_id == 0  # TODO
        self.trials[trial_id].params[param_name] = param_value

    def set_trial_value(self, study_id, trial_id, value):
        assert study_id == 0  # TODO
        self.trials[trial_id].value = value

    def set_trial_intermediate_value(self, study_id, trial_id, step, intermediate_value):
        assert study_id == 0  # TODO
        self.trials[trial_id].intermediate_values[step] = intermediate_value

    def set_trial_system_attr(self, study_id, trial_id, attr_name, attr_value):
        assert study_id == 0  # TODO
        self.trials[trial_id].system_attrs[attr_name] = attr_value

    def get_trial(self, study_id, trial_id):
        assert study_id == 0  # TODO
        return copy.deepcopy(self.trials[trial_id])

    def get_all_trials(self, study_id):
        assert study_id == 0  # TODO
        return copy.deepcopy(self.trials)
