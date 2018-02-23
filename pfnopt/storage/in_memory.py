import copy
from typing import Dict, Any  # NOQA
from typing import List  # NOQA

from pfnopt import distributions  # NOQA
from pfnopt import trial

from pfnopt.storage import base


class InMemoryStorage(base.BaseStorage):

    def __init__(self):
        # type: () -> None
        self.study_attrs = {}  # type: Dict[str, Any]
        self.trials = []  # type: List[trial.Trial]
        self.param_distribution = {}  # type: Dict[str, distributions.BaseDistribution]

    def create_new_trial_id(self, study_id):
        # type: (int) -> int
        assert study_id == 0  # TODO(Akiba)
        trial_id = len(self.trials)
        self.trials.append(trial.Trial(trial_id))
        return trial_id

    def set_study_param_distribution(self, study_id, param_name, distribution):
        # type: (int, str, distributions.BaseDistribution) -> None
        assert study_id == 0
        self.param_distribution[param_name] = distribution

    def set_trial_state(self, study_id, trial_id, state):
        # type: (int, int, trial.State) -> None
        assert study_id == 0  # TODO(Akiba)
        self.trials[trial_id].state = state

    def set_trial_param(self, study_id, trial_id, param_name, param_value_in_internal_repr):
        # type: (int, int, str, float) -> None
        assert study_id == 0  # TODO(Akiba)
        self.trials[trial_id].params_in_internal_repr[param_name] = param_value_in_internal_repr

        distribution = self.param_distribution[param_name]
        param_value_actual = distribution.to_external_repr(param_value_in_internal_repr)
        self.trials[trial_id].params[param_name] = param_value_actual

    def set_trial_value(self, study_id, trial_id, value):
        # type: (int, int, float) -> None
        assert study_id == 0  # TODO(Akiba)
        self.trials[trial_id].value = value

    def set_trial_intermediate_value(self, study_id, trial_id, step, intermediate_value):
        # type: (int, int, int, float) -> None
        assert study_id == 0  # TODO(Akiba)
        self.trials[trial_id].intermediate_values[step] = intermediate_value

    def set_trial_system_attr(self, study_id, trial_id, attr_name, attr_value):
        # type: (int, int, str, Any) -> None
        assert study_id == 0  # TODO(Akiba)
        self.trials[trial_id].system_attrs[attr_name] = attr_value

    def get_trial(self, study_id, trial_id):
        # type: (int, int) -> trial.Trial
        assert study_id == 0  # TODO(Akiba)
        return copy.deepcopy(self.trials[trial_id])

    def get_all_trials(self, study_id):
        # type: (int) -> List[trial.Trial]
        assert study_id == 0  # TODO(Akiba)
        return copy.deepcopy(self.trials)
