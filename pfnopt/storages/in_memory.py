import copy
import threading
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import List  # NOQA

from pfnopt import distributions  # NOQA
from pfnopt.study_summary import StudySummary
from pfnopt.study_summary import StudySystemAttributes
from pfnopt import trial

from pfnopt.storages import base


IN_MEMORY_STORAGE_STUDY_ID = 0
IN_MEMORY_STORAGE_STUDY_UUID = '00000000-0000-0000-0000-000000000000'


class InMemoryStorage(base.BaseStorage):

    def __init__(self):
        # type: () -> None
        self.trials = []  # type: List[trial.Trial]
        self.param_distribution = {}  # type: Dict[str, distributions.BaseDistribution]
        self.study_user_attrs = {}  # type: Dict[str, Any]
        self.study_system_attrs = StudySystemAttributes(None, None)  # type: StudySystemAttributes

        self._lock = threading.Lock()

    def __getstate__(self):
        # type: () -> Dict[Any, Any]
        state = self.__dict__.copy()
        del state['_lock']
        return state

    def __setstate__(self, state):
        # type: (Dict[Any, Any]) -> None
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def create_new_study_id(self):
        # type: () -> int

        return IN_MEMORY_STORAGE_STUDY_ID  # TODO(akiba)

    def get_study_id_from_uuid(self, study_uuid):
        # type: (str) -> int

        self._check_study_uuid(study_uuid)
        return IN_MEMORY_STORAGE_STUDY_ID

    def get_study_uuid_from_id(self, study_id):
        # type: (int) -> str

        self._check_study_id(study_id)
        return IN_MEMORY_STORAGE_STUDY_UUID

    def set_study_system_attrs(self, study_id, system_attrs):
        # type: (int, StudySystemAttributes) -> None

        with self._lock:
            self.study_system_attrs = system_attrs

    def get_study_system_attrs(self, study_id):
        # type: (int) -> StudySystemAttributes

        with self._lock:
            return copy.deepcopy(self.study_system_attrs)

    def set_study_user_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        with self._lock:
            self.study_user_attrs[key] = value

    def get_study_user_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        with self._lock:
            return copy.deepcopy(self.study_user_attrs)

    def get_all_study_summaries(self):
        # type: () -> List[StudySummary]

        with self._lock:
            summary = StudySummary(
                study_id=IN_MEMORY_STORAGE_STUDY_ID,
                study_uuid=IN_MEMORY_STORAGE_STUDY_UUID,
                system_attrs=self.study_system_attrs,
                user_attrs=copy.deepcopy(self.study_user_attrs),
                n_trials=len(self.trials)
            )
            return [summary]

    def create_new_trial_id(self, study_id):
        # type: (int) -> int

        self._check_study_id(study_id)
        with self._lock:
            trial_id = len(self.trials)
            self.trials.append(
                trial.Trial(
                    trial_id=trial_id,
                    state=trial.State.RUNNING,
                    params={},
                    system_attrs=trial.SystemAttributes(
                        datetime_start=None,
                        datetime_complete=None),
                    user_attrs={},
                    value=None,
                    intermediate_values={},
                    params_in_internal_repr={}
                )
            )
        return trial_id

    def set_trial_param_distribution(self, trial_id, param_name, distribution):
        # type: (int, str, distributions.BaseDistribution) -> None

        with self._lock:
            if param_name in self.param_distribution:
                distributions.check_distribution_compatibility(
                    self.param_distribution[param_name], distribution)
            self.param_distribution[param_name] = distribution

    def set_trial_state(self, trial_id, state):
        # type: (int, trial.State) -> None

        with self._lock:
            self.trials[trial_id] = self.trials[trial_id]._replace(state=state)

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
            self.trials[trial_id] = self.trials[trial_id]._replace(value=value)

    def set_trial_intermediate_value(self, trial_id, step, intermediate_value):
        # type: (int, int, float) -> None

        with self._lock:
            self.trials[trial_id].intermediate_values[step] = intermediate_value

    def set_trial_system_attrs(self, trial_id, system_attrs):
        # type: (int, trial.SystemAttributes) -> None

        with self._lock:
            self.trials[trial_id] = self.trials[trial_id]._replace(system_attrs=system_attrs)

    def set_trial_user_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        with self._lock:
            self.trials[trial_id].user_attrs[key] = value

    def get_trial(self, trial_id):
        # type: (int) -> trial.Trial

        with self._lock:
            return copy.deepcopy(self.trials[trial_id])

    def get_all_trials(self, study_id):
        # type: (int) -> List[trial.Trial]

        self._check_study_id(study_id)
        with self._lock:
            return copy.deepcopy(self.trials)

    def _check_study_id(self, study_id):
        # type: (int) -> None

        if study_id != IN_MEMORY_STORAGE_STUDY_ID:
            raise ValueError('study_id is supposed to be {} in {}.'.format(
                IN_MEMORY_STORAGE_STUDY_ID, self.__class__.__name__))

    def _check_study_uuid(self, study_uuid):
        # type: (str) -> None

        if study_uuid != IN_MEMORY_STORAGE_STUDY_UUID:
            raise ValueError('study_uuid is supposed to be {} in {}.'.format(
                IN_MEMORY_STORAGE_STUDY_UUID, self.__class__.__name__))
