import copy
import threading
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from optuna import distributions
from optuna.storages import base
from optuna.storages.rdb.storage import RDBStorage
from optuna.study import StudyDirection
from optuna.study import StudySummary
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class _TrialUpdate:
    def __init__(self) -> None:
        self.state = None  # type: Optional[TrialState]
        self.value = None  # type: Optional[float]
        self.values = dict()  # type: Dict[int, float]
        self.user_attrs = dict()  # type: Dict[str, Any]
        self.system_attrs = dict()  # type: Dict[str, Any]
        self.params = dict()  # type: Dict[str, Any]
        self.distributions = dict()  # type: Dict[str, distributions.BaseDistribution]


class _StudyInfo:
    def __init__(self) -> None:
        self.trials = dict()  # type: Dict[int, FrozenTrial]
        self.updates = dict()  # type: Dict[int, _TrialUpdate]
        self.param_distribution = {}  # type: Dict[str, distributions.BaseDistribution]


class _CachedStorage(base.BaseStorage):
    """A wrapper class of storage backends.

    This class is used in :func:`~optuna.get_storage` function and automatically
    wraps :class:`~optuna.storages.RDBStorage` class.

    Args:
        storage:
            :class:`~optuna.storages.BaseStorage` class instance to wrap.
    """

    def __init__(self, backend: RDBStorage) -> None:
        self._backend = backend
        self._studies = {}  # type: Dict[int, _StudyInfo]
        self._trial_id_to_study_id_and_number = dict()  # type: Dict[int, Tuple[int, int]]
        self._lock = threading.RLock()

    def __getstate__(self) -> Dict[Any, Any]:
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: Dict[Any, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def create_new_study(self, study_name: Optional[str] = None) -> int:

        study_id = self._backend.create_new_study(study_name)
        with self._lock:
            self._studies[study_id] = _StudyInfo()
        return study_id

    def delete_study(self, study_id: int) -> None:

        with self._lock:
            if study_id in self._studies:
                for trial_id in self._studies[study_id].trials:
                    trial_ids_to_delete = []
                    if trial_id in self._trial_id_to_study_id_and_number:
                        trial_ids_to_delete.append(trial_id)
                    for trial_id in trial_ids_to_delete:
                        del self._trial_id_to_study_id_and_number[trial_id]
                del self._studies[study_id]

        self._backend.delete_study(study_id)

    def set_study_direction(self, study_id: int, direction: StudyDirection) -> None:

        self._backend.set_study_direction(study_id, direction)

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:

        self._backend.set_study_user_attr(study_id, key, value)

    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:

        self._backend.set_study_system_attr(study_id, key, value)

    def get_study_id_from_name(self, study_name: str) -> int:

        return self._backend.get_study_id_from_name(study_name)

    def get_study_id_from_trial_id(self, trial_id: int) -> int:

        return self._backend.get_study_id_from_trial_id(trial_id)

    def get_study_name_from_id(self, study_id: int) -> str:

        return self._backend.get_study_name_from_id(study_id)

    def get_study_direction(self, study_id: int) -> StudyDirection:

        return self._backend.get_study_direction(study_id)

    def get_study_user_attrs(self, study_id: int) -> Dict[str, Any]:

        return self._backend.get_study_user_attrs(study_id)

    def get_study_system_attrs(self, study_id: int) -> Dict[str, Any]:

        return self._backend.get_study_system_attrs(study_id)

    def get_all_study_summaries(self) -> List[StudySummary]:

        return self._backend.get_all_study_summaries()

    def create_new_trial(self, study_id: int, template_trial: Optional[FrozenTrial] = None) -> int:

        trial_id, frozen_trial = self._backend._create_new_trial_with_trial(
            study_id, template_trial
        )
        with self._lock:
            if study_id not in self._studies:
                self._studies[study_id] = _StudyInfo()
            self._trial_id_to_study_id_and_number[trial_id] = study_id, frozen_trial.number
            if frozen_trial.state == TrialState.RUNNING:
                self._studies[study_id].trials[frozen_trial.number] = frozen_trial
            return trial_id

    def set_trial_state(self, trial_id: int, state: TrialState) -> bool:

        with self._lock:
            cached_trial = self._get_cached_trial(trial_id)
            if cached_trial is not None:
                updates = self._get_updates(trial_id)
                cached_trial.state = state
                updates.state = state
                ret = self._flush_trial(trial_id)
                if cached_trial.state != TrialState.RUNNING:
                    study_id, number = self._trial_id_to_study_id_and_number[trial_id]
                    del self._studies[study_id].trials[number]
                return ret

        return self._backend.set_trial_state(trial_id, state)

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: distributions.BaseDistribution,
    ) -> bool:

        with self._lock:
            cached_trial = self._get_cached_trial(trial_id)
            if cached_trial is not None:
                updates = self._get_updates(trial_id)
                study_id, _ = self._trial_id_to_study_id_and_number[trial_id]
                cached_dist = self._studies[study_id].param_distribution.get(param_name, None)
                if cached_dist:
                    distributions.check_distribution_compatibility(cached_dist, distribution)
                else:
                    self._backend._check_or_set_param_distribution(
                        trial_id, param_name, param_value_internal, distribution
                    )
                    self._studies[study_id].param_distribution[param_name] = distribution
                if param_name in cached_trial.params:
                    return False
                params = copy.copy(cached_trial.params)
                params[param_name] = distribution.to_external_repr(param_value_internal)
                cached_trial.params = params
                dists = copy.copy(cached_trial.distributions)
                dists[param_name] = distribution
                cached_trial.distributions = dists
                updates.params[param_name] = param_value_internal
                updates.distributions[param_name] = distribution
                return True

        return self._backend.set_trial_param(
            trial_id, param_name, param_value_internal, distribution
        )

    def get_trial_number_from_id(self, trial_id: int) -> int:

        return self.get_trial(trial_id).number

    def get_best_trial(self, study_id: int) -> FrozenTrial:

        return self._backend.get_best_trial(study_id)

    def get_trial_param(self, trial_id: int, param_name: str) -> float:

        trial = self.get_trial(trial_id)
        return trial.distributions[param_name].to_internal_repr(trial.params[param_name])

    def set_trial_value(self, trial_id: int, value: float) -> None:

        with self._lock:
            cached_trial = self._get_cached_trial(trial_id)
            if cached_trial is not None:
                updates = self._get_updates(trial_id)
                cached_trial.value = value
                updates.value = value
                self._flush_trial(trial_id)
                return

        self._backend._update_trial(trial_id, value=value)

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> bool:

        with self._lock:
            cached_trial = self._get_cached_trial(trial_id)
            if cached_trial is not None:
                updates = self._get_updates(trial_id)
                if step in cached_trial.intermediate_values:
                    return False
                values = copy.copy(cached_trial.intermediate_values)
                values[step] = intermediate_value
                cached_trial.intermediate_values = values
                updates.values[step] = intermediate_value
                self._flush_trial(trial_id)
                return True

        return self._backend.set_trial_intermediate_value(trial_id, step, intermediate_value)

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:

        with self._lock:
            cached_trial = self._get_cached_trial(trial_id)
            if cached_trial is not None:
                updates = self._get_updates(trial_id)
                attrs = copy.copy(cached_trial.user_attrs)
                attrs[key] = value
                cached_trial.user_attrs = attrs
                updates.user_attrs[key] = value
                self._flush_trial(trial_id)
                return

        self._backend._update_trial(trial_id, user_attrs={key: value})

    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:

        with self._lock:
            cached_trial = self._get_cached_trial(trial_id)
            if cached_trial is not None:
                updates = self._get_updates(trial_id)
                attrs = copy.copy(cached_trial.system_attrs)
                attrs[key] = value
                cached_trial.system_attrs = attrs
                updates.system_attrs[key] = value
                self._flush_trial(trial_id)
                return

        self._backend._update_trial(trial_id, system_attrs={key: value})

    def _get_cached_trial(self, trial_id: int) -> Optional[FrozenTrial]:
        if trial_id not in self._trial_id_to_study_id_and_number:
            return None
        study_id, number = self._trial_id_to_study_id_and_number[trial_id]
        return self._studies[study_id].trials.get(number, None)

    def _get_updates(self, trial_id: int) -> _TrialUpdate:
        study_id, number = self._trial_id_to_study_id_and_number[trial_id]
        updates = self._studies[study_id].updates.get(number, None)
        if updates is not None:
            return updates
        updates = _TrialUpdate()
        self._studies[study_id].updates[number] = updates
        return updates

    def get_trial(self, trial_id: int) -> FrozenTrial:

        with self._lock:
            trial = self._get_cached_trial(trial_id)
            if trial is not None:
                return copy.deepcopy(trial)

        return self._backend.get_trial(trial_id)

    def get_all_trials(self, study_id: int, deepcopy: bool = True) -> List[FrozenTrial]:

        with self._lock:
            trials = self._backend.get_all_trials(study_id, deepcopy=False)
            if study_id in self._studies:
                for key, trial in self._studies[study_id].trials.items():
                    trials[trial.number] = trial
            return copy.deepcopy(trials) if deepcopy else trials

    def get_n_trials(self, study_id: int, state: Optional[TrialState] = None) -> int:

        return self._backend.get_n_trials(study_id, state)

    def _flush_trial(self, trial_id: int) -> bool:
        if trial_id not in self._trial_id_to_study_id_and_number:
            # The trial has not been managed by this class.
            return True
        study_id, number = self._trial_id_to_study_id_and_number[trial_id]
        study = self._studies[study_id]
        updates = study.updates.get(number, None)
        if updates is None:
            # The trial is up-to-date.
            return True
        del study.updates[number]
        return self._backend._update_trial(
            trial_id=trial_id,
            value=updates.value,
            values=updates.values,
            state=updates.state,
            params=updates.params,
            dists=updates.distributions,
            user_attrs=updates.user_attrs,
            system_attrs=updates.system_attrs,
        )
