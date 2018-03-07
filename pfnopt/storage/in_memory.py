import copy
import threading
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA

from pfnopt import distributions  # NOQA
from pfnopt import trial

from pfnopt.storage import base


IN_MEMORY_STORAGE_STUDY_ID = 0
IN_MEMORY_STORAGE_STUDY_UUID = '00000000-0000-0000-0000-000000000000'


class InMemoryStorage(base.BaseStorage):

    def __init__(self):
        # type: () -> None
        self.study_attrs = {}  # type: Dict[str, Any]
        self.trials = []  # type: List[trial.Trial]
        self.param_distribution = {}  # type: Dict[str, distributions.BaseDistribution]

        self._lock = threading.Lock()

    def create_new_study_id(self):
        return IN_MEMORY_STORAGE_STUDY_ID  # TODO(akiba)

    def get_study_id(self, study_uuid):
        # type: (str) -> Optional[int]
        return IN_MEMORY_STORAGE_STUDY_ID

    def get_study_uuid(self, study_id):
        # type: (int) -> Optional[str]
        return IN_MEMORY_STORAGE_STUDY_UUID

    def create_new_trial_id(self, study_id):
        # type: (int) -> int
        assert study_id == IN_MEMORY_STORAGE_STUDY_ID  # TODO(Akiba)
        with self._lock:
            trial_id = len(self.trials)
            self.trials.append(trial.Trial(trial_id))
        return trial_id

    def set_study_param_distribution(self, study_id, param_name, distribution):
        # type: (int, str, distributions.BaseDistribution) -> None
        assert study_id == IN_MEMORY_STORAGE_STUDY_ID
        with self._lock:
            self.param_distribution[param_name] = distribution

    def set_trial_state(self, trial_id, state):
        # type: (int, trial.State) -> None
        with self._lock:
            self.trials[trial_id].state = state

    def set_trial_param(self, trial_id, param_name, param_value_in_internal_repr):
        # type: (int, str, float) -> None
        with self._lock:
            self.trials[trial_id].params_in_internal_repr[param_name] = \
                param_value_in_internal_repr
            distribution = self.param_distribution[param_name]
            param_value_actual = distribution.to_external_repr(param_value_in_internal_repr)
            self.trials[trial_id].params[param_name] = param_value_actual

    def set_trial_value(self, trial_id, value):
        # type: (int, float) -> None
        with self._lock:
            self.trials[trial_id].value = value

    def set_trial_intermediate_value(self, trial_id, step, intermediate_value):
        # type: (int, int, float) -> None
        with self._lock:
            self.trials[trial_id].intermediate_values[step] = intermediate_value

    def set_trial_system_attrs(self, trial_id, system_attrs):
        # type: (int, trial.SystemAttributes) -> None
        with self._lock:
            self.trials[trial_id].system_attrs = system_attrs

    def get_trial(self, trial_id):
        # type: (int) -> trial.Trial
        with self._lock:
            return copy.deepcopy(self.trials[trial_id])

    def get_all_trials(self, study_id):
        # type: (int) -> List[trial.Trial]
        assert study_id == IN_MEMORY_STORAGE_STUDY_ID  # TODO(Akiba)
        with self._lock:
            return copy.deepcopy(self.trials)
