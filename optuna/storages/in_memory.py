import copy
from datetime import datetime
import threading

from optuna import distributions  # NOQA
from optuna.storages import base
from optuna.storages.base import DEFAULT_STUDY_NAME_PREFIX
from optuna import structs
from optuna import types

if types.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA

IN_MEMORY_STORAGE_STUDY_ID = 0
IN_MEMORY_STORAGE_STUDY_UUID = '00000000-0000-0000-0000-000000000000'


class InMemoryStorage(base.BaseStorage):
    """Storage class that stores data in memory of the Python process.

    This class is not supposed to be directly accessed by library users.
    """

    def __init__(self):
        # type: () -> None
        self.trials = []  # type: List[structs.FrozenTrial]
        self.param_distribution = {}  # type: Dict[str, distributions.BaseDistribution]
        self.direction = structs.StudyDirection.NOT_SET
        self.study_user_attrs = {}  # type: Dict[str, Any]
        self.study_system_attrs = {}  # type: Dict[str, Any]
        self.study_name = DEFAULT_STUDY_NAME_PREFIX + IN_MEMORY_STORAGE_STUDY_UUID  # type: str

        self._lock = threading.RLock()

    def __getstate__(self):
        # type: () -> Dict[Any, Any]
        state = self.__dict__.copy()
        del state['_lock']
        return state

    def __setstate__(self, state):
        # type: (Dict[Any, Any]) -> None
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def create_new_study_id(self, study_name=None):
        # type: (Optional[str]) -> int

        if study_name is not None:
            self.study_name = study_name

        return IN_MEMORY_STORAGE_STUDY_ID  # TODO(akiba)

    def set_study_direction(self, study_id, direction):
        # type: (int, structs.StudyDirection) -> None

        with self._lock:
            if self.direction != structs.StudyDirection.NOT_SET and self.direction != direction:
                raise ValueError('Cannot overwrite study direction from {} to {}.'.format(
                    self.direction, direction))
            self.direction = direction

    def set_study_user_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        with self._lock:
            self.study_user_attrs[key] = value

    def set_study_system_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        with self._lock:
            self.study_system_attrs[key] = value

    def get_study_id_from_name(self, study_name):
        # type: (str) -> int

        if study_name != self.study_name:
            raise ValueError("No such study {}.".format(study_name))

        return IN_MEMORY_STORAGE_STUDY_ID

    def get_study_id_from_trial_id(self, trial_id):
        # type: (int) -> int

        return IN_MEMORY_STORAGE_STUDY_ID

    def get_study_name_from_id(self, study_id):
        # type: (int) -> str

        self._check_study_id(study_id)
        return self.study_name

    def get_study_direction(self, study_id):
        # type: (int) -> structs.StudyDirection

        return self.direction

    def get_study_user_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        with self._lock:
            return copy.deepcopy(self.study_user_attrs)

    def get_study_system_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        with self._lock:
            return copy.deepcopy(self.study_system_attrs)

    def get_all_study_summaries(self):
        # type: () -> List[structs.StudySummary]

        best_trial = None
        n_complete_trials = len([t for t in self.trials if t.state == structs.TrialState.COMPLETE])
        if n_complete_trials > 0:
            best_trial = self.get_best_trial(IN_MEMORY_STORAGE_STUDY_ID)

        datetime_start = None
        if len(self.trials) > 0:
            datetime_start = min([t.datetime_start for t in self.trials])

        return [
            structs.StudySummary(
                study_id=IN_MEMORY_STORAGE_STUDY_ID,
                study_name=self.study_name,
                direction=self.direction,
                best_trial=best_trial,
                user_attrs=copy.deepcopy(self.study_user_attrs),
                system_attrs=copy.deepcopy(self.study_system_attrs),
                n_trials=len(self.trials),
                datetime_start=datetime_start)
        ]

    def create_new_trial_id(self, study_id):
        # type: (int) -> int

        self._check_study_id(study_id)
        with self._lock:
            trial_id = len(self.trials)
            self.trials.append(
                structs.FrozenTrial(
                    number=trial_id,
                    state=structs.TrialState.RUNNING,
                    params={},
                    distributions={},
                    user_attrs={},
                    system_attrs={'_number': trial_id},
                    value=None,
                    intermediate_values={},
                    params_in_internal_repr={},
                    datetime_start=datetime.now(),
                    datetime_complete=None,
                    trial_id=trial_id))
        return trial_id

    def set_trial_state(self, trial_id, state):
        # type: (int, structs.TrialState) -> None

        with self._lock:
            self.check_trial_is_updatable(trial_id, self.trials[trial_id].state)

            self.trials[trial_id] = self.trials[trial_id]._replace(state=state)
            if state.is_finished():
                self.trials[trial_id] = \
                    self.trials[trial_id]._replace(datetime_complete=datetime.now())

    def set_trial_param(self, trial_id, param_name, param_value_internal, distribution):
        # type: (int, str, float, distributions.BaseDistribution) -> bool

        with self._lock:
            self.check_trial_is_updatable(trial_id, self.trials[trial_id].state)

            # Check param distribution compatibility with previous trial(s).
            if param_name in self.param_distribution:
                distributions.check_distribution_compatibility(self.param_distribution[param_name],
                                                               distribution)

            # Check param has not been set; otherwise, return False.
            if param_name in self.trials[trial_id].params:
                return False

            # Set param distribution.
            self.param_distribution[param_name] = distribution

            # Set param.
            self.trials[trial_id].params_in_internal_repr[param_name] = param_value_internal
            self.trials[trial_id].params[param_name] = distribution.to_external_repr(
                param_value_internal)
            self.trials[trial_id].distributions[param_name] = distribution

            return True

    def get_trial_number_from_id(self, trial_id):
        # type: (int) -> int

        return trial_id

    def get_trial_param(self, trial_id, param_name):
        # type: (int, str) -> float

        return self.trials[trial_id].params_in_internal_repr[param_name]

    def set_trial_value(self, trial_id, value):
        # type: (int, float) -> None

        with self._lock:
            self.check_trial_is_updatable(trial_id, self.trials[trial_id].state)

            self.trials[trial_id] = self.trials[trial_id]._replace(value=value)

    def set_trial_intermediate_value(self, trial_id, step, intermediate_value):
        # type: (int, int, float) -> bool

        with self._lock:
            self.check_trial_is_updatable(trial_id, self.trials[trial_id].state)

            values = self.trials[trial_id].intermediate_values
            if step in values:
                return False

            values[step] = intermediate_value

            return True

    def set_trial_user_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        with self._lock:
            self.check_trial_is_updatable(trial_id, self.trials[trial_id].state)

            self.trials[trial_id].user_attrs[key] = value

    def set_trial_system_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        with self._lock:
            self.check_trial_is_updatable(trial_id, self.trials[trial_id].state)

            self.trials[trial_id].system_attrs[key] = value

    def get_trial(self, trial_id):
        # type: (int) -> structs.FrozenTrial

        with self._lock:
            return copy.deepcopy(self.trials[trial_id])

    def get_all_trials(self, study_id):
        # type: (int) -> List[structs.FrozenTrial]

        self._check_study_id(study_id)
        with self._lock:
            return copy.deepcopy(self.trials)

    def get_n_trials(self, study_id, state=None):
        # type: (int, Optional[structs.TrialState]) -> int

        self._check_study_id(study_id)
        if state is None:
            return len(self.trials)

        return len([t for t in self.trials if t.state == state])

    def _check_study_id(self, study_id):
        # type: (int) -> None

        if study_id != IN_MEMORY_STORAGE_STUDY_ID:
            raise ValueError('study_id is supposed to be {} in {}.'.format(
                IN_MEMORY_STORAGE_STUDY_ID, self.__class__.__name__))
