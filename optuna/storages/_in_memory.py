from collections import Counter
import copy
from datetime import datetime
import threading
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
import uuid

from optuna import distributions  # NOQA
from optuna.exceptions import DuplicatedStudyError
from optuna.storages._base import _BackEnd
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.storages._cached_storage import _CachedStorage
from optuna.study import StudyDirection
from optuna.study import StudySummary
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class InMemoryStorage(_CachedStorage):
    """Storage class that stores data in memory of the Python process.

    This class is not supposed to be directly accessed by library users.
    """

    def __init__(self) -> None:
        super(InMemoryStorage, self).__init__(InMemoryBackend())


class InMemoryBackend(_BackEnd):
    """This class is supposed to be used with :class:`_CachedStorage`."""

    def __init__(self) -> None:
        self._study_name_to_id = {}  # type: Dict[str, int]
        self._studies = {}  # type: Dict[int, _StudyInfo]
        self._unfinished_trials = {}  # type: Dict[int, Tuple[FrozenTrial, int]]

        self._max_study_id = -1
        self._max_trial_id = -1

        self._lock = threading.RLock()

    def __getstate__(self) -> Dict[Any, Any]:
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: Dict[Any, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def create_new_study(self, study_name: Optional[str] = None) -> int:

        with self._lock:
            study_id = self._max_study_id + 1
            self._max_study_id += 1

            if study_name is not None:
                if study_name in self._study_name_to_id:
                    raise DuplicatedStudyError
            else:
                study_uuid = str(uuid.uuid4())
                study_name = DEFAULT_STUDY_NAME_PREFIX + study_uuid
            self._studies[study_id] = _StudyInfo(study_name)
            self._study_name_to_id[study_name] = study_id

            return study_id

    def delete_study(self, study_id: int) -> None:

        with self._lock:
            self._check_study_id(study_id)

            study_name = self._studies[study_id].name
            del self._study_name_to_id[study_name]
            del self._studies[study_id]

    def set_study_direction(self, study_id: int, direction: StudyDirection) -> None:

        with self._lock:
            self._check_study_id(study_id)

            study = self._studies[study_id]
            if study.direction != StudyDirection.NOT_SET and study.direction != direction:
                raise ValueError(
                    "Cannot overwrite study direction from {} to {}.".format(
                        study.direction, direction
                    )
                )
            study.direction = direction

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:

        with self._lock:
            self._check_study_id(study_id)

            self._studies[study_id].user_attrs[key] = value

    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:

        with self._lock:
            self._check_study_id(study_id)

            self._studies[study_id].system_attrs[key] = value

    def get_study_id_from_name(self, study_name: str) -> int:

        with self._lock:
            if study_name not in self._study_name_to_id:
                raise KeyError("No such study {}.".format(study_name))

            return self._study_name_to_id[study_name]

    def get_study_id_from_trial_id(self, trial_id: int) -> int:
        self._check_trial_id(trial_id)
        raise AssertionError(self._error_msg())

    def get_study_name_from_id(self, study_id: int) -> str:

        with self._lock:
            self._check_study_id(study_id)
            return self._studies[study_id].name

    def get_study_direction(self, study_id: int) -> StudyDirection:

        with self._lock:
            self._check_study_id(study_id)
            return self._studies[study_id].direction

    def get_study_user_attrs(self, study_id: int) -> Dict[str, Any]:

        with self._lock:
            self._check_study_id(study_id)
            return self._studies[study_id].user_attrs

    def get_study_system_attrs(self, study_id: int) -> Dict[str, Any]:

        with self._lock:
            self._check_study_id(study_id)
            return self._studies[study_id].system_attrs

    def get_all_study_summaries(self) -> List[StudySummary]:

        with self._lock:
            return [self._build_study_summary(study_id) for study_id in self._studies.keys()]

    def _build_study_summary(self, study_id: int) -> StudySummary:
        study = self._studies[study_id]
        return StudySummary(
            study_name=study.name,
            direction=study.direction,
            best_trial=copy.deepcopy(study.best_trial),
            user_attrs=copy.deepcopy(study.user_attrs),
            system_attrs=copy.deepcopy(study.system_attrs),
            n_trials=study.n_trials,
            datetime_start=study.datetime_start,
            study_id=study_id,
        )

    def create_new_trial(self, study_id: int, template_trial: Optional[FrozenTrial] = None) -> int:
        raise AssertionError(self._error_msg())

    def _create_new_trial(
        self, study_id: int, template_trial: Optional[FrozenTrial] = None
    ) -> FrozenTrial:

        with self._lock:
            self._check_study_id(study_id)

            if template_trial is None:
                trial = self._create_unfinished_trial()
            else:
                trial = copy.deepcopy(template_trial)

            study = self._studies[study_id]
            trial_id = self._max_trial_id + 1
            self._max_trial_id += 1
            trial._trial_id = trial_id
            trial.number = study.n_trials
            study.n_trials_per_state[trial.state] += 1
            if not trial.state.is_finished():
                self._unfinished_trials[trial_id] = trial, study_id
            if study.datetime_start is None:
                study.datetime_start = trial.datetime_start
            else:
                study.datetime_start = min(trial.datetime_start, study.datetime_start)
            if trial.state == TrialState.COMPLETE:
                self._update_best_value(trial, study_id)
            return trial

    @staticmethod
    def _create_unfinished_trial() -> FrozenTrial:
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

    def set_trial_state(self, trial_id: int, state: TrialState) -> bool:
        # This function is called only when waking ``WAITING`` trials up.
        self._check_trial_id(trial_id)
        if self._unfinished_trials[trial_id][0].state != TrialState.WAITING:
            raise AssertionError(self._error_msg())
        if state != TrialState.RUNNING:
            raise AssertionError(self._error_msg())
        self._unfinished_trials[trial_id][0].state = state
        return True

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: distributions.BaseDistribution,
    ) -> None:
        self._check_trial_id(trial_id)
        raise AssertionError(self._error_msg())

    def get_trial_number_from_id(self, trial_id: int) -> int:
        self._check_trial_id(trial_id)
        raise AssertionError(self._error_msg())

    def get_best_trial(self, study_id: int) -> FrozenTrial:
        with self._lock:
            self._check_study_id(study_id)

            best_trial = self._studies[study_id].best_trial
            if best_trial is None:
                raise ValueError("No trials are completed yet.")
            return best_trial

    def get_trial_param(self, trial_id: int, param_name: str) -> float:
        self._check_trial_id(trial_id)
        raise AssertionError(self._error_msg())

    def set_trial_value(self, trial_id: int, value: float) -> None:
        self._check_trial_id(trial_id)
        raise AssertionError(self._error_msg())

    def _update_trial(
        self,
        trial_id: int,
        state: Optional[TrialState] = None,
        value: Optional[float] = None,
        intermediate_values: Optional[Dict[int, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        distributions_: Optional[Dict[str, distributions.BaseDistribution]] = None,
        user_attrs: Optional[Dict[str, Any]] = None,
        system_attrs: Optional[Dict[str, Any]] = None,
        datetime_complete: Optional[datetime] = None,
    ) -> bool:
        trial, study_id = self._unfinished_trials[trial_id]
        if state is not None:
            study = self._studies[study_id]
            study.n_trials_per_state[trial.state] -= 1
            study.n_trials_per_state[state] += 1
            trial.state = state
        if value is not None:
            trial.value = value
        if intermediate_values:
            trial.intermediate_values.update(intermediate_values)
        if params and distributions_:
            trial.params.update(
                {key: distributions_[key].to_external_repr(val) for key, val in params.items()}
            )
            trial.distributions.update(distributions_)
        if user_attrs:
            trial.user_attrs.update(user_attrs)
        if system_attrs:
            trial.system_attrs.update(system_attrs)
        if datetime_complete is not None:
            trial.datetime_complete = datetime_complete
        if state is not None and state.is_finished():
            del self._unfinished_trials[trial_id]
        if state is not None and state == TrialState.COMPLETE:
            self._update_best_value(trial, study_id)
        return True

    def _update_best_value(self, trial: FrozenTrial, study_id: int) -> None:
        study = self._studies[study_id]
        if study.best_trial is None or study.best_trial.value is None:
            study.best_trial = trial
            return
        prev_best_value = study.best_trial.value
        assert prev_best_value is not None  # For mypy checks.
        if study.direction == StudyDirection.MAXIMIZE:
            if trial.value is not None and not (prev_best_value >= trial.value):
                study.best_trial = trial
        elif study.direction == StudyDirection.MINIMIZE:
            if trial.value is not None and not (prev_best_value <= trial.value):
                study.best_trial = trial
        else:
            raise AssertionError(self._error_msg())

    def _check_and_set_param_distribution(
        self,
        study_id: int,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: distributions.BaseDistribution,
    ) -> None:
        self._update_trial(
            trial_id,
            params={param_name: param_value_internal},
            distributions_={param_name: distribution},
        )

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        self._check_trial_id(trial_id)
        raise AssertionError(self._error_msg())

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        self._check_trial_id(trial_id)
        raise AssertionError(self._error_msg())

    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:
        self._check_trial_id(trial_id)
        raise AssertionError(self._error_msg())

    def get_trial(self, trial_id: int) -> FrozenTrial:
        self._check_trial_id(trial_id)
        return self._unfinished_trials[trial_id][0]

    def get_all_trials(self, study_id: int, deepcopy: bool = True) -> List[FrozenTrial]:
        self._check_study_id(study_id)
        raise AssertionError(self._error_msg())

    def get_n_trials(self, study_id: int, state: Optional[TrialState] = None) -> int:
        if state is None:
            return self._studies[study_id].n_trials
        else:
            return self._studies[study_id].n_trials_per_state[state]

    def read_trials_from_remote_storage(self, study_id: int) -> None:
        self._check_study_id(study_id)

    def _check_study_id(self, study_id: int) -> None:

        print(study_id, self._studies.keys())
        if study_id not in self._studies:
            raise KeyError("No study with study_id {} exists.".format(study_id))

    def _check_trial_id(self, trial_id: int) -> None:

        if trial_id not in self._unfinished_trials:
            raise KeyError("No trial with trial_id {} exists.".format(trial_id))

    def _get_trials(self, study_id: int, excluded_trial_ids: Set[int]) -> List[FrozenTrial]:
        self._check_study_id(study_id)
        return []

    @staticmethod
    def _error_msg() -> str:
        return "Internal error. Please report a bug to Optuna."


class _StudyInfo:
    def __init__(self, name: str) -> None:
        self.n_trials_per_state = Counter()  # type: Counter
        self.param_distribution = {}  # type: Dict[str, distributions.BaseDistribution]
        self.user_attrs = {}  # type: Dict[str, Any]
        self.system_attrs = {}  # type: Dict[str, Any]
        self.name = name  # type: str
        self.direction = StudyDirection.NOT_SET
        self.best_trial = None  # type: Optional[FrozenTrial]
        self.datetime_start = None  # type: Optional[datetime]

    @property
    def n_trials(self) -> int:
        return sum(self.n_trials_per_state.values())
