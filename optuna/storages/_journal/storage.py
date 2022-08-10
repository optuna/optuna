import copy
import datetime
import enum
import threading
from typing import Any
from typing import Container
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
import uuid

import optuna
from optuna._experimental import experimental_class
from optuna.distributions import BaseDistribution
from optuna.distributions import check_distribution_compatibility
from optuna.distributions import distribution_to_json
from optuna.distributions import json_to_distribution
from optuna.exceptions import DuplicatedStudyError
from optuna.storages import BaseStorage
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.storages._journal.base import BaseJournalLogStorage
from optuna.study._frozen import FrozenStudy
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = optuna.logging.get_logger(__name__)

NOT_FOUND_MSG = "Record does not exist."


class JournalOperation(enum.IntEnum):
    CREATE_STUDY = 0
    DELETE_STUDY = 1
    SET_STUDY_USER_ATTR = 2
    SET_STUDY_SYSTEM_ATTR = 3
    SET_STUDY_DIRECTIONS = 4
    CREATE_TRIAL = 5
    SET_TRIAL_PARAM = 6
    SET_TRIAL_STATE_VALUES = 7
    SET_TRIAL_INTERMEDIATE_VALUE = 8
    SET_TRIAL_USER_ATTR = 9
    SET_TRIAL_SYSTEM_ATTR = 10


def datetime_from_isoformat(datetime_str: str) -> datetime.datetime:
    return datetime.datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%f")


@experimental_class("3.1.0")
class JournalStorage(BaseStorage):
    """Storage class for Journal storage backend.

    Note that library users can instantiate this class, but the attributes
    provided by this class are not supposed to be directly accessed by them.

    Journal storage writes a record of every operation to the database as it is executed and
    at the same time, keeps a latest snapshot of the database in-memory. If the database crashes
    for any reason, the storage can re-establish the contents in memory by replaying the
    operations stored from the beginning.

    Journal storage has several benefits over the conventional value logging storages.
    1. The number of IOs can be reduced because of larger granularity of logs.
    2. Journal storage has simpler backend API than value logging storage.
    3. Journal storage keeps a snapshot in-memory so no need to add more cache.

    Example:

        .. code::

            storage = JournalStorage(JournalFileStorage("./log_file"))

            study = optuna.create_study(storage=storage)

    """

    threading.Lock()

    def __init__(self, log_storage: BaseJournalLogStorage) -> None:
        self._pid = str(uuid.uuid4())
        self._log_number_read: int = 0
        self._backend = log_storage

        # In-memory replayed results
        self._studies: Dict[int, FrozenStudy] = dict()
        self._trials: Dict[int, FrozenTrial] = dict()
        self._study_id_to_trial_ids: Dict[int, List[int]] = dict()
        self._trial_id_to_study_id: Dict[int, int] = dict()
        self._next_study_id: int = 0
        self._trial_ids_owned_by_this_process: List[int] = []

        self._thread_lock = threading.Lock()

    def _write_log(self, op_code: int, extra_fields: Dict[str, Any]) -> None:
        self._backend.append_logs([{"op_code": op_code, "pid": self._pid, **extra_fields}])

    def _raise_if_log_issued_by_pid(self, log: Dict[str, Any], err: Exception) -> None:
        if log["pid"] == self._pid:
            raise err

    def _apply_create_study(self, log: Dict[str, Any]) -> None:
        study_name = log["study_name"]

        if study_name in [s.study_name for s in self._studies.values()]:
            self._raise_if_log_issued_by_pid(
                log,
                DuplicatedStudyError(
                    "Another study with name '{}' already exists. "
                    "Please specify a different name, or reuse the existing one "
                    "by setting `load_if_exists` (for Python API) or "
                    "`--skip-if-exists` flag (for CLI).".format(study_name)
                ),
            )
            return

        study_id = self._next_study_id
        self._next_study_id += 1

        self._studies[study_id] = FrozenStudy(
            study_name=study_name,
            direction=StudyDirection.NOT_SET,
            user_attrs={},
            system_attrs={},
            study_id=study_id,
        )
        self._study_id_to_trial_ids[study_id] = []

    def _study_exists(self, study_id: int, log: Dict[str, Any]) -> bool:
        if study_id not in self._studies:
            self._raise_if_log_issued_by_pid(log, KeyError(NOT_FOUND_MSG))
            return False
        return True

    def _apply_delete_study(self, log: Dict[str, Any]) -> None:
        study_id = log["study_id"]

        if self._study_exists(study_id, log):
            fs = self._studies.pop(study_id)
            assert fs._study_id == study_id

    def _apply_set_study_user_attr(self, log: Dict[str, Any]) -> None:
        study_id = log["study_id"]

        if self._study_exists(study_id, log):
            assert len(log["user_attr"]) == 1
            self._studies[study_id].user_attrs.update(log["user_attr"])

    def _apply_set_study_system_attr(self, log: Dict[str, Any]) -> None:
        study_id = log["study_id"]

        if self._study_exists(study_id, log):
            assert len(log["system_attr"]) == 1
            self._studies[study_id].system_attrs.update(log["system_attr"])

    def _apply_set_study_directions(self, log: Dict[str, Any]) -> None:
        study_id = log["study_id"]

        if not self._study_exists(study_id, log):
            return

        directions = [StudyDirection(d) for d in log["directions"]]

        current_directions = self._studies[study_id]._directions
        if current_directions[0] != StudyDirection.NOT_SET and current_directions != directions:
            self._raise_if_log_issued_by_pid(
                log,
                ValueError(
                    "Cannot overwrite study direction from {} to {}.".format(
                        current_directions, directions
                    )
                ),
            )
            return

        self._studies[study_id]._directions = [StudyDirection(d) for d in directions]

    def _apply_create_trial(self, log: Dict[str, Any]) -> None:
        study_id = log["study_id"]

        if not self._study_exists(study_id, log):
            return

        trial_id = len(self._trials)
        distributions = (
            {}
            if "distributions" not in log
            else {k: json_to_distribution(v) for k, v in log["distributions"].items()}
        )
        params = (
            {}
            if "params" not in log
            else {
                k: distributions[k].to_external_repr(param) for k, param in log["params"].items()
            }
        )
        if log["datetime_start"] is not None:
            datetime_start = datetime_from_isoformat(log["datetime_start"])
        else:
            datetime_start = None
        if "datetime_complete" in log:
            datetime_complete = datetime_from_isoformat(log["datetime_complete"])
        else:
            datetime_complete = None

        self._trials[trial_id] = FrozenTrial(
            trial_id=trial_id,
            number=len(self._study_id_to_trial_ids[study_id]),
            state=TrialState(log.get("state", TrialState.RUNNING.value)),
            params=params,
            distributions=distributions,
            user_attrs=log.get("user_attrs", {}),
            system_attrs=log.get("system_attrs", {}),
            value=log.get("value", None),
            intermediate_values={int(k): v for k, v in log.get("intermediate_values", {}).items()},
            datetime_start=datetime_start,
            datetime_complete=datetime_complete,
            values=log.get("values", None),
        )

        self._study_id_to_trial_ids[study_id].append(trial_id)
        self._trial_id_to_study_id[trial_id] = study_id

        if log["pid"] == self._pid:
            self._trial_ids_owned_by_this_process.append(trial_id)

    def _trial_exists_and_updatable(self, trial_id: int, log: Dict[str, Any]) -> bool:
        if trial_id not in self._trials:
            self._raise_if_log_issued_by_pid(log, KeyError(NOT_FOUND_MSG))
            return False
        elif self._trials[trial_id].state.is_finished():
            self._raise_if_log_issued_by_pid(
                log,
                RuntimeError(
                    "Trial#{} has already finished and can not be updated.".format(
                        self._trials[trial_id].number
                    )
                ),
            )
            return False
        else:
            return True

    def _apply_set_trial_param(self, log: Dict[str, Any]) -> None:
        trial_id = log["trial_id"]

        if not self._trial_exists_and_updatable(trial_id, log):
            return

        param_name = log["param_name"]
        param_value_internal = log["param_value_internal"]
        distribution = json_to_distribution(log["distribution"])

        study_id = self._trial_id_to_study_id[trial_id]

        for prev_trial_id in self._study_id_to_trial_ids[study_id]:
            prev_trial = self._trials[prev_trial_id]
            if param_name in prev_trial.params.keys():
                try:
                    check_distribution_compatibility(
                        prev_trial.distributions[param_name], distribution
                    )
                except Exception as e:
                    self._raise_if_log_issued_by_pid(log, e)
                    return
                break

        trial = copy.copy(self._trials[trial_id])
        trial.params = {
            **copy.copy(trial.params),
            param_name: distribution.to_external_repr(param_value_internal),
        }
        trial.distributions = {**copy.copy(trial.distributions), param_name: distribution}
        self._trials[trial_id] = trial

    def _apply_set_trial_state_values(self, log: Dict[str, Any]) -> None:
        trial_id = log["trial_id"]

        if not self._trial_exists_and_updatable(trial_id, log):
            return

        state = TrialState(log["state"])
        if state == self._trials[trial_id].state and state == TrialState.RUNNING:
            return

        trial = copy.copy(self._trials[trial_id])
        if state == TrialState.RUNNING:
            trial.datetime_start = datetime_from_isoformat(log["datetime_start"])
        if state.is_finished():
            trial.datetime_complete = datetime_from_isoformat(log["datetime_complete"])
        trial.state = state
        if log["values"] is not None:
            trial.values = log["values"]

        self._trials[trial_id] = trial
        return

    def _apply_set_trial_intermediate_value(self, log: Dict[str, Any]) -> None:
        trial_id = log["trial_id"]

        if self._trial_exists_and_updatable(trial_id, log):
            trial = copy.copy(self._trials[trial_id])
            trial.intermediate_values = {
                **copy.copy(trial.intermediate_values),
                log["step"]: log["intermediate_value"],
            }
            self._trials[trial_id] = trial

    def _apply_set_trial_user_attr(self, log: Dict[str, Any]) -> None:
        trial_id = log["trial_id"]

        if self._trial_exists_and_updatable(trial_id, log):
            assert len(log["user_attr"]) == 1
            trial = copy.copy(self._trials[trial_id])
            trial.user_attrs = {**copy.copy(trial.user_attrs), **log["user_attr"]}
            self._trials[trial_id] = trial

    def _apply_set_trial_system_attr(self, log: Dict[str, Any]) -> None:
        trial_id = log["trial_id"]

        if self._trial_exists_and_updatable(trial_id, log):
            assert len(log["system_attr"]) == 1
            trial = copy.copy(self._trials[trial_id])
            trial.system_attrs = {
                **copy.copy(trial.system_attrs),
                **log["system_attr"],
            }
            self._trials[trial_id] = trial

    def _apply_log(self, log: Dict[str, Any]) -> None:
        op = log["op_code"]
        if op == JournalOperation.CREATE_STUDY:
            self._apply_create_study(log)
        elif op == JournalOperation.DELETE_STUDY:
            self._apply_delete_study(log)
        elif op == JournalOperation.SET_STUDY_USER_ATTR:
            self._apply_set_study_user_attr(log)
        elif op == JournalOperation.SET_STUDY_SYSTEM_ATTR:
            self._apply_set_study_system_attr(log)
        elif op == JournalOperation.SET_STUDY_DIRECTIONS:
            self._apply_set_study_directions(log)
        elif op == JournalOperation.CREATE_TRIAL:
            self._apply_create_trial(log)
        elif op == JournalOperation.SET_TRIAL_PARAM:
            self._apply_set_trial_param(log)
        elif op == JournalOperation.SET_TRIAL_STATE_VALUES:
            self._apply_set_trial_state_values(log)
        elif op == JournalOperation.SET_TRIAL_INTERMEDIATE_VALUE:
            self._apply_set_trial_intermediate_value(log)
        elif op == JournalOperation.SET_TRIAL_USER_ATTR:
            self._apply_set_trial_user_attr(log)
        elif op == JournalOperation.SET_TRIAL_SYSTEM_ATTR:
            self._apply_set_trial_system_attr(log)
        else:
            assert False, "Should not reach."

    def _sync_with_backend(self) -> None:
        logs = self._backend.read_logs(self._log_number_read)
        for log in logs:
            self._log_number_read += 1
            self._apply_log(log)

    # Basic study manipulation

    def create_new_study(self, study_name: Optional[str] = None) -> int:
        study_name = study_name or DEFAULT_STUDY_NAME_PREFIX + str(uuid.uuid4())
        with self._thread_lock:
            self._write_log(JournalOperation.CREATE_STUDY, {"study_name": study_name})
            self._sync_with_backend()

            for frozen_study in self._studies.values():
                if frozen_study.study_name == study_name:
                    _logger.info("A new study created in Journal with name: {}".format(study_name))
                    return frozen_study._study_id
            assert False, "Should not reach."

    def delete_study(self, study_id: int) -> None:
        with self._thread_lock:
            self._write_log(JournalOperation.DELETE_STUDY, {"study_id": study_id})
            self._sync_with_backend()

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        log: Dict[str, Any] = {"study_id": study_id, "user_attr": {key: value}}
        with self._thread_lock:
            self._write_log(JournalOperation.SET_STUDY_USER_ATTR, log)
            self._sync_with_backend()

    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:
        log: Dict[str, Any] = {"study_id": study_id, "system_attr": {key: value}}
        with self._thread_lock:
            self._write_log(JournalOperation.SET_STUDY_SYSTEM_ATTR, log)
            self._sync_with_backend()

    def set_study_directions(self, study_id: int, directions: Sequence[StudyDirection]) -> None:
        log: Dict[str, Any] = {"study_id": study_id, "directions": directions}
        with self._thread_lock:
            self._write_log(JournalOperation.SET_STUDY_DIRECTIONS, log)
            self._sync_with_backend()

    # Basic study access

    def get_study_id_from_name(self, study_name: str) -> int:
        with self._thread_lock:
            self._sync_with_backend()
            frozen_study = [fs for fs in self._studies.values() if fs.study_name == study_name]
            if len(frozen_study) == 0:
                raise KeyError(NOT_FOUND_MSG)
            assert len(frozen_study) == 1
            return frozen_study[0]._study_id

    def _raise_if_not_found_in_studies(self, study_id: int) -> None:
        if study_id not in self._studies.keys():
            raise KeyError(NOT_FOUND_MSG)

    def get_study_name_from_id(self, study_id: int) -> str:
        with self._thread_lock:
            self._sync_with_backend()
            self._raise_if_not_found_in_studies(study_id)
            return self._studies[study_id].study_name

    def get_study_directions(self, study_id: int) -> List[StudyDirection]:
        with self._thread_lock:
            self._sync_with_backend()
            self._raise_if_not_found_in_studies(study_id)
            return self._studies[study_id].directions

    def get_study_user_attrs(self, study_id: int) -> Dict[str, Any]:
        with self._thread_lock:
            self._sync_with_backend()
            self._raise_if_not_found_in_studies(study_id)
            return self._studies[study_id].user_attrs

    def get_study_system_attrs(self, study_id: int) -> Dict[str, Any]:
        with self._thread_lock:
            self._sync_with_backend()
            self._raise_if_not_found_in_studies(study_id)
            return self._studies[study_id].system_attrs

    def get_all_studies(self) -> List[FrozenStudy]:
        with self._thread_lock:
            self._sync_with_backend()
            return copy.deepcopy(list(self._studies.values()))

    # Basic trial manipulation
    def create_new_trial(self, study_id: int, template_trial: Optional[FrozenTrial] = None) -> int:
        log: Dict[str, Any] = {
            "study_id": study_id,
            "datetime_start": datetime.datetime.now().isoformat(),
        }

        if template_trial:
            log["state"] = template_trial.state
            if template_trial.values is not None and len(template_trial.values) > 1:
                log["value"] = None
                log["values"] = template_trial.values
            else:
                log["value"] = template_trial.value
                log["values"] = None
            if template_trial.datetime_start:
                log["datetime_start"] = template_trial.datetime_start.isoformat()
            else:
                log["datetime_start"] = None
            if template_trial.datetime_complete:
                log["datetime_complete"] = template_trial.datetime_complete.isoformat()

            log["distributions"] = {
                k: distribution_to_json(dist) for k, dist in template_trial.distributions.items()
            }
            log["params"] = {
                k: template_trial.distributions[k].to_internal_repr(param)
                for k, param in template_trial.params.items()
            }
            log["user_attrs"] = template_trial.user_attrs
            log["system_attrs"] = template_trial.system_attrs
            log["intermediate_values"] = template_trial.intermediate_values

        with self._thread_lock:
            self._write_log(JournalOperation.CREATE_TRIAL, log)
            self._sync_with_backend()
            return self._trial_ids_owned_by_this_process[-1]

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
    ) -> None:
        log: Dict[str, Any] = {
            "trial_id": trial_id,
            "param_name": param_name,
            "param_value_internal": param_value_internal,
            "distribution": distribution_to_json(distribution),
        }

        with self._thread_lock:
            self._write_log(JournalOperation.SET_TRIAL_PARAM, log)
            self._sync_with_backend()

    def get_trial_id_from_study_id_trial_number(self, study_id: int, trial_number: int) -> int:
        with self._thread_lock:
            self._sync_with_backend()
            if len(self._study_id_to_trial_ids[study_id]) <= trial_number:
                raise KeyError(
                    "No trial with trial number {} exists in study with study_id {}.".format(
                        trial_number, study_id
                    )
                )
            return self._study_id_to_trial_ids[study_id][trial_number]

    def set_trial_state_values(
        self, trial_id: int, state: TrialState, values: Optional[Sequence[float]] = None
    ) -> bool:
        log: Dict[str, Any] = {
            "trial_id": trial_id,
            "state": state,
            "values": values,
        }

        if state == TrialState.RUNNING:
            log["datetime_start"] = datetime.datetime.now().isoformat()
        elif state.is_finished():
            log["datetime_complete"] = datetime.datetime.now().isoformat()

        with self._thread_lock:
            self._write_log(JournalOperation.SET_TRIAL_STATE_VALUES, log)
            self._sync_with_backend()

            if (
                state == TrialState.RUNNING
                and trial_id not in self._trial_ids_owned_by_this_process
            ):
                return False
            else:
                return True

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        log: Dict[str, Any] = {
            "trial_id": trial_id,
            "step": step,
            "intermediate_value": intermediate_value,
        }

        with self._thread_lock:
            self._write_log(JournalOperation.SET_TRIAL_INTERMEDIATE_VALUE, log)
            self._sync_with_backend()

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        log: Dict[str, Any] = {
            "trial_id": trial_id,
            "user_attr": {key: value},
        }

        with self._thread_lock:
            self._write_log(JournalOperation.SET_TRIAL_USER_ATTR, log)
            self._sync_with_backend()

    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:
        log: Dict[str, Any] = {
            "trial_id": trial_id,
            "system_attr": {key: value},
        }

        with self._thread_lock:
            self._write_log(JournalOperation.SET_TRIAL_SYSTEM_ATTR, log)
            self._sync_with_backend()

    # Basic trial access

    def get_trial(self, trial_id: int) -> FrozenTrial:
        with self._thread_lock:
            self._sync_with_backend()
            frozen_trial = [
                trial for trial in self._trials.values() if trial._trial_id == trial_id
            ]
            if len(frozen_trial) != 1:
                raise KeyError(NOT_FOUND_MSG)
            else:
                return frozen_trial[0]

    def get_all_trials(
        self,
        study_id: int,
        deepcopy: bool = True,
        states: Optional[Container[TrialState]] = None,
    ) -> List[FrozenTrial]:
        with self._thread_lock:
            self._sync_with_backend()
            self._raise_if_not_found_in_studies(study_id)

            frozen_trials = []

            for trial_id in self._study_id_to_trial_ids[study_id]:
                trial = self._trials[trial_id]
                if states is None or trial.state in states:
                    if deepcopy:
                        frozen_trials.append(copy.deepcopy(trial))
                    else:
                        frozen_trials.append(trial)

            return frozen_trials
