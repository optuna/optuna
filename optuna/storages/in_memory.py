import copy
from collections import defaultdict
from datetime import datetime
import threading
import uuid

from optuna import distributions  # NOQA
from optuna.exceptions import DuplicatedStudyError
from optuna.storages import base
from optuna.storages.base import DEFAULT_STUDY_NAME_PREFIX
from optuna.study import StudyDirection
from optuna.study import StudySummary
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import DefaultDict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA


class InMemoryStorage(base.BaseStorage):
    """Storage class that stores data in memory of the Python process.

    This class is not supposed to be directly accessed by library users.
    """

    def __init__(self):
        # type: () -> None
        self._trial_id_to_study_id_and_number = {}  # type: Dict[int, Tuple[int, int]]
        self._trials = []  # type: List[Optional[FrozenTrial]]
        self._study_trials = defaultdict(list)  # type: DefaultDict[int, List[int]]
        self._param_distribution = defaultdict(dict)  # type: DefaultDict[int, Dict[str, distributions.BaseDistribution]]
        self._direction = defaultdict(lambda: StudyDirection.NOT_SET)  # type: DefaultDict[int, StudyDirection]
        self._study_user_attrs = defaultdict(dict)  # type: DefaultDict[int, Dict[str, Any]]
        self._study_system_attrs = defaultdict(dict)  # type: DefaultDict[int, Dict[str, Any]]
        self._study_name = {}  # type: Dict[int, str]
        self._study_name_to_id = {}  # type: Dict[str, int]
        self._best_trial_id = defaultdict(lambda: None)  # type: DefaultDict[int, Optional[int]]

        self._max_study_id = -1

        self._lock = threading.RLock()

    def __getstate__(self):
        # type: () -> Dict[Any, Any]
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state):
        # type: (Dict[Any, Any]) -> None
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def create_new_study(self, study_name=None):
        # type: (Optional[str]) -> int

        study_id = self._max_study_id + 1
        self._max_study_id += 1

        if study_name is not None:
            if study_name in self._study_name_to_id:
                raise DuplicatedStudyError
            self._study_name[study_id] = study_name
            self._study_name_to_id[study_name] = study_id
        else:
            study_uuid = str(uuid.uuid4())
            study_name = DEFAULT_STUDY_NAME_PREFIX + study_uuid
            self._study_name[study_id] = study_name
            self._study_name_to_id[study_name] = study_id

        return study_id

    def delete_study(self, study_id):
        # type: (int) -> None

        self._check_study_id(study_id)

        with self._lock:
            for trial_id in self._study_trials[study_id]:
                self._trials[trial_id] = None
                del self._trial_id_to_study_id_and_number[trial_id]
            del self._study_trials[study_id]
            if study_id in self._best_trial_id:
                del self._best_trial_id[study_id]
            if study_id in self._param_distribution:
                del self._param_distribution[study_id]
            if study_id in self._direction:
                del self._direction[study_id]
            if study_id in self._study_user_attrs:
                del self._study_user_attrs[study_id]
            if study_id in self._study_system_attrs:
                del self._study_system_attrs[study_id]
            del self._study_name[study_id]

    def set_study_direction(self, study_id, direction):
        # type: (int, StudyDirection) -> None

        self._check_study_id(study_id)

        with self._lock:
            if self._direction[study_id] != StudyDirection.NOT_SET and self._direction[study_id] != direction:
                raise ValueError(
                    "Cannot overwrite study direction from {} to {}.".format(
                        self._direction[study_id], direction
                    )
                )
            self._direction[study_id] = direction

    def set_study_user_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        self._check_study_id(study_id)

        with self._lock:
            self._study_user_attrs[study_id][key] = value

    def set_study_system_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        self._check_study_id(study_id)

        with self._lock:
            self._study_system_attrs[study_id][key] = value

    def get_study_id_from_name(self, study_name):
        # type: (str) -> int

        if study_name not in self._study_name_to_id:
            raise KeyError("No such study {}.".format(study_name))

        return self._study_name_to_id[study_name]

    def get_study_id_from_trial_id(self, trial_id):
        # type: (int) -> int

        self._check_trial_id(trial_id)

        return self._trial_id_to_study_id_and_number[trial_id][0]

    def get_study_name_from_id(self, study_id):
        # type: (int) -> str

        self._check_study_id(study_id)
        return self._study_name[study_id]

    def get_study_direction(self, study_id):
        # type: (int) -> StudyDirection

        self._check_study_id(study_id)
        return self._direction[study_id]

    def get_study_user_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        self._check_study_id(study_id)
        with self._lock:
            return self._study_user_attrs[study_id]

    def get_study_system_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        self._check_study_id(study_id)
        with self._lock:
            return self._study_system_attrs[study_id]

    def get_all_study_summaries(self):
        # type: () -> List[StudySummary]

        return [self._build_study_summary(study_id) for study_id in self._study_name.keys()]

    def _build_study_summary(self, study_id: int) -> StudySummary:
        best_trial_id = self._best_trial_id[study_id]
        return StudySummary(
            study_name=self._study_name[study_id],
            direction=self._direction[study_id],
            best_trial=self._trials[best_trial_id] if best_trial_id is not None else None,
            user_attrs=copy.copy(self._study_user_attrs[study_id]),
            system_attrs=copy.copy(self._study_system_attrs[study_id]),
            n_trials=len(self._study_trials[study_id]),
            datetime_start=min(
                [
                    trial.datetime_start
                    for trial in self.get_all_trials(study_id, deepcopy=False)
                ]
            ) if self._study_trials[study_id] else None,
            study_id=study_id,
        )

    def create_new_trial(self, study_id, template_trial=None):
        # type: (int, Optional[FrozenTrial]) -> int

        self._check_study_id(study_id)

        if template_trial is None:
            trial = self._create_running_trial()
        else:
            trial = copy.deepcopy(template_trial)

        with self._lock:
            trial_id = len(self._trials)
            trial.number = len(self._study_trials[study_id])
            trial._trial_id = trial_id
            self._trials.append(trial)
            self._update_cache(trial_id, study_id)
            self._trial_id_to_study_id_and_number[trial_id] = (study_id, trial.number)
            self._study_trials[study_id].append(trial_id)
        return trial_id

    @staticmethod
    def _create_running_trial():
        # type: () -> FrozenTrial

        return FrozenTrial(
            trial_id=-1,  # dummy value.
            number=-1,  # dummy value.
            state=TrialState.RUNNING,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            value=None,
            intermediate_values={},
            datetime_start=datetime.now(),
            datetime_complete=None,
        )

    def set_trial_state(self, trial_id, state):
        # type: (int, TrialState) -> bool

        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)

        with self._lock:
            trial = copy.copy(trial)
            self.check_trial_is_updatable(trial_id, trial.state)

            if state == TrialState.RUNNING and trial.state != TrialState.WAITING:
                return False

            trial.state = state
            if state.is_finished():
                trial.datetime_complete = datetime.now()
                self._trials[trial_id] = trial
                study_id = self._trial_id_to_study_id_and_number[trial_id][0]
                self._update_cache(trial_id, study_id)
            else:
                self._trials[trial_id] = trial

        return True

    def set_trial_param(self, trial_id, param_name, param_value_internal, distribution):
        # type: (int, str, float, distributions.BaseDistribution) -> bool

        trial = self.get_trial(trial_id)

        with self._lock:
            self.check_trial_is_updatable(trial_id, trial.state)

            study_id = self._trial_id_to_study_id_and_number[trial_id][0]
            # Check param distribution compatibility with previous trial(s).
            if param_name in self._param_distribution[study_id]:
                distributions.check_distribution_compatibility(
                    self._param_distribution[study_id][param_name], distribution
                )

            # Check param has not been set; otherwise, return False.
            if param_name in trial.params:
                return False

            # Set param distribution.
            self._param_distribution[study_id][param_name] = distribution

            # Set param.
            trial = copy.copy(trial)
            trial.params = copy.copy(trial.params)
            trial.params[param_name] = distribution.to_external_repr(param_value_internal)
            trial.distributions = copy.copy(trial.distributions)
            trial.distributions[param_name] = distribution
            self._trials[trial_id] = trial

            return True

    def get_trial_number_from_id(self, trial_id):
        # type: (int) -> int

        self._check_trial_id(trial_id)

        return self._trial_id_to_study_id_and_number[trial_id][1]

    def get_best_trial(self, study_id):
        # type: (int) -> FrozenTrial

        self._check_study_id(study_id)

        best_trial_id = self._best_trial_id[study_id]
        if best_trial_id is None:
            raise ValueError("No trials are completed yet.")
        return self.get_trial(best_trial_id)

    def get_trial_param(self, trial_id, param_name):
        # type: (int, str) -> float

        trial = self.get_trial(trial_id)

        distribution = trial.distributions[param_name]
        return distribution.to_internal_repr(trial.params[param_name])

    def set_trial_value(self, trial_id, value):
        # type: (int, float) -> None

        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)

        with self._lock:
            trial = copy.copy(trial)
            self.check_trial_is_updatable(trial_id, trial.state)

            trial.value = value
            self._trials[trial_id] = trial

    def _update_cache(self, trial_id: int, study_id: int) -> None:

        trial = self.get_trial(trial_id)

        if trial.state != TrialState.COMPLETE:
            return

        best_trial_id = self._best_trial_id[study_id]
        if best_trial_id is None:
            self._best_trial_id[study_id] = trial_id
            return
        best_trial = self._trials[best_trial_id]
        assert best_trial is not None
        best_value = best_trial.value
        new_value = trial.value
        if best_value is None:
            self._best_trial_id[study_id] = trial_id
            return
        # Complete trials do not have `None` values.
        assert new_value is not None

        if self.get_study_direction(study_id) == StudyDirection.MAXIMIZE:
            if best_value < new_value:
                self._best_trial_id[study_id] = trial_id
            return
        if best_value > new_value:
            self._best_trial_id[study_id] = trial_id
        return

    def set_trial_intermediate_value(self, trial_id, step, intermediate_value):
        # type: (int, int, float) -> bool

        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)

        with self._lock:
            self.check_trial_is_updatable(trial_id, trial.state)

            trial = copy.copy(trial)
            values = copy.copy(trial.intermediate_values)
            if step in values:
                return False

            values[step] = intermediate_value
            trial.intermediate_values = values
            self._trials[trial_id] = trial

            return True

    def set_trial_user_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        self._check_trial_id(trial_id)
        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)

        with self._lock:
            self.check_trial_is_updatable(trial_id, trial.state)

            trial = copy.copy(trial)
            trial.user_attrs = copy.copy(trial.user_attrs)
            trial.user_attrs[key] = value
            self._trials[trial_id] = trial

    def set_trial_system_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)

        with self._lock:
            self.check_trial_is_updatable(trial_id, trial.state)

            trial = copy.copy(trial)
            trial.system_attrs = copy.copy(trial.system_attrs)
            trial.system_attrs[key] = value
            self._trials[trial_id] = trial

    def get_trial(self, trial_id):
        # type: (int) -> FrozenTrial

        self._check_trial_id(trial_id)
        trial = self._trials[trial_id]
        assert trial is not None
        return trial

    def get_all_trials(self, study_id, deepcopy=True):
        # type: (int, bool) -> List[FrozenTrial]

        # TODO(ytsmiling) Rewrite the whole trial management logic for faster get_all_trials.

        self._check_study_id(study_id)
        trials = [self._trials[tid] for tid in self._study_trials[study_id]]
        with self._lock:
            if deepcopy:
                return [copy.deepcopy(trial) for trial in trials if trial is not None]
            else:
                return [trial for trial in trials if trial is not None]

    def get_n_trials(self, study_id, state=None):
        # type: (int, Optional[TrialState]) -> int

        self._check_study_id(study_id)
        if state is None:
            return len(self._study_trials[study_id])

        return len([0 for trial in self.get_all_trials(study_id, deepcopy=False) if trial.state == state])

    def _check_study_id(self, study_id):
        # type: (int) -> None

        with self._lock:
            if study_id not in self._study_name:
                raise KeyError("No study with study_id {} exists.".format(study_id))

    def _check_trial_id(self, trial_id: int) -> None:

        with self._lock:
            if trial_id >= len(self._trials) or self._trials[trial_id] is None:
                raise KeyError("No trial with trial_id {} exists.".format(trial_id))
