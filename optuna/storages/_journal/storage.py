import datetime
import enum
import os
import socket
import threading
from typing import Any
from typing import cast
from typing import Container
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import uuid

from optuna.distributions import BaseDistribution
from optuna.distributions import distribution_to_json
from optuna.distributions import json_to_distribution
from optuna.exceptions import DuplicatedStudyError
from optuna.storages import BaseStorage
from optuna.storages._journal.file import FileStorage
from optuna.study._frozen import FrozenStudy
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


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


class JournalStorage(BaseStorage):
    """Base class for storages.

    This class is not supposed to be directly accessed by library users.

    A storage class abstracts a backend database and provides library internal interfaces to
    read/write histories of studies and trials.

    **Thread safety**

    A storage class can be shared among multiple threads, and must therefore be thread-safe.
    It must guarantee that return values such as `FrozenTrial`s are never modified.
    A storage class can assume that return values are never modified by its user.
    When a user modifies a return value from a storage class, the internal state of the storage
    may become inconsistent. Consequences are undefined.

    **Ownership of RUNNING trials**

    Trials in finished states are not allowed to be modified.
    Trials in the WAITING state are not allowed to be modified except for the `state` field.
    A storage class can assume that each RUNNING trial is only modified from a single process.
    When a user modifies a RUNNING trial from multiple processes, the internal state of the storage
    may become inconsistent. Consequences are undefined.
    A storage class is not intended for inter-process communication.
    Consequently, users using optuna with MPI or other multi-process programs must make sure that
    only one process is used to access the optuna interface.

    **Consistency models**

    A storage class must support the monotonic-reads consistency model, that is, if a
    process reads data `X`, any successive reads on data `X` cannot return older values.
    It must support read-your-writes, that is, if a process writes to data `X`,
    any successive reads on data `X` from the same process must read the written
    value or one of the more recent values.

    **Stronger consistency requirements for special data**

    Under a multi-worker setting, a storage class must return the latest values of any attributes
    of a study and a trial. Generally, typical storages naturally hold this requirement. However,
    :class:`~optuna.storages._CachedStorage` does not, so we introduce the
    `read_trials_from_remote_storage(study_id)` method in the class. The detailed explanation how
    :class:`~optuna.storages._CachedStorage` aquires this requirement, is available at
    the docstring.

    .. note::

        These attribute behaviors may become user customizable in the future.

    **Data persistence**

    A storage class does not guarantee that write operations are logged into a persistent
    storage, even when write methods succeed.
    Thus, when process failure occurs, some writes might be lost.
    As exceptions, when a persistent storage is available, any writes on any attributes
    of `Study` and writes on `state` of `Trial` are guaranteed to be persistent.
    Additionally, any preceding writes on any attributes of `Trial` are guaranteed to
    be written into a persistent storage before writes on `state` of `Trial` succeed.
    The same applies for `param`, `user_attrs', 'system_attrs' and 'intermediate_values`
    attributes.

    .. note::

        These attribute behaviors may become user customizable in the future.
    """

    def __init__(self) -> None:
        self._host_name = socket.gethostname()
        self._host_ip = socket.gethostbyname(self._host_name)
        self._p_id = os.getpid()
        self._me = self._host_name + "--" + self._host_ip + "--" + str(self._p_id)

        self._log_number_read: int = 0
        self._backend = FileStorage("operation_logs")

        # study-id - FrozenStudy dict (thread-unsafe)
        self._studies: Dict[int, FrozenStudy] = dict()

        # trial-id - FrozenTrial dict (thread-unsafe)
        self._trials: Dict[int, FrozenTrial] = dict()
        self._trial_ids_created_by_me: Dict[str, int] = dict()

        # replay result
        self._replay_result_of_logs_created_by_me: Dict[str, Any] = dict()

        # study-id - [trial-id] dict (thread-unsafe)
        self._study_id_to_trial_ids: Dict[int, List[int]] = dict()

        # (thread-safe)
        self._thread_local = threading.local()
        self._thread_local._buffered_logs = []
        self._thread_local._id = threading.get_ident()

        # Lock for controlling multiple threads
        self._thread_lock = threading.Lock()

    def _create_operation_log(self, op_code: JournalOperation) -> Dict[str, Any]:
        return {
            "log_id": str(uuid.uuid4()) + "--" + str(self._thread_local._id),
            "op_code": op_code,
            "me": self._me,
        }

    def _buffer_log(self, log: Dict[str, Any]) -> None:
        self._thread_local._buffered_logs.append(log)

    def _flush_logs(self) -> None:
        self._backend.append_logs(self._thread_local._buffered_logs)
        self._thread_local._buffered_logs = []

    def _push_log_replay_result(self, log: Dict[str, Any], result: Any) -> None:
        if log["me"] == self._me:
            self._replay_result_of_logs_created_by_me[log["log_id"]] = result

    def _pop_log_replay_result(self, log: Dict[str, Any]) -> Any:
        assert log["me"] == self._me
        return self._replay_result_of_logs_created_by_me.pop(log["log_id"])

    def _apply_log(self, log: Dict[str, Any]) -> None:
        op = log["op_code"]
        if op == JournalOperation.CREATE_STUDY:
            study_name = log["study_name"]

            if study_name in [s.study_name for s in self._studies.values()]:
                self._push_log_replay_result(log, DuplicatedStudyError)
                return

            study_id = len(self._studies)

            fs = FrozenStudy(
                study_name=study_name,
                direction=None,
                user_attrs={},
                system_attrs={},
                study_id=study_id,
            )

            self._studies[study_id] = fs

            self._push_log_replay_result(log, study_id)

        elif op == JournalOperation.DELETE_STUDY:
            study_id = log["study_id"]

            if study_id not in self._studies.keys():
                self._push_log_replay_result(log, KeyError)
                return

            fs = self._studies.pop(study_id)

            assert fs._study_id == study_id

            self._push_log_replay_result(log, None)

        elif op == JournalOperation.SET_STUDY_USER_ATTR:
            study_id = log["study_id"]

            if study_id not in self._studies.keys():
                self._push_log_replay_result(log, KeyError)
                return

            user_attr = "user_attr"
            assert len(log[user_attr].items()) == 1

            ((key, value),) = log[user_attr].items()

            self._studies[study_id].user_attrs[key] = value

            self._push_log_replay_result(log, None)

        elif op == JournalOperation.SET_STUDY_SYSTEM_ATTR:
            study_id = log["study_id"]

            if study_id not in self._studies.keys():
                self._push_log_replay_result(log, KeyError)
                return

            system_attr = "system_attr"
            assert len(log[system_attr].items()) == 1

            ((key, value),) = log[system_attr].items()

            self._studies[study_id].system_attrs[key] = value

            self._push_log_replay_result(log, None)

        elif op == JournalOperation.SET_STUDY_DIRECTIONS:
            study_id = log["study_id"]

            if study_id not in self._studies.keys():
                self._push_log_replay_result(log, KeyError)
                return

            directions = log["directions"]
            self._studies[study_id]._directions = directions

            self._push_log_replay_result(log, None)

        elif op == JournalOperation.CREATE_TRIAL:
            study_id = log["study_id"]

            if study_id not in self._studies.keys():
                self._push_log_replay_result(log, KeyError)
                return

            trial_id = len(self._trials)

            user_attrs = {}
            system_attrs = {}
            intermediate_values = {}
            state = TrialState.RUNNING

            if log["has_template_trial"]:
                user_attrs = log["user_attrs"]
                system_attrs = log["system_attrs"]
                intermediate_values = log["intermediate_values"]
                state = log["state"]

            self._trials[trial_id] = FrozenTrial(
                trial_id=trial_id,
                number=-1,
                state=state,
                params={},
                distributions={},
                user_attrs=user_attrs,
                system_attrs=system_attrs,
                value=None,
                intermediate_values=intermediate_values,
                datetime_start=datetime.datetime.now(),
                datetime_complete=None,
            )

            self._study_id_to_trial_ids.setdefault(study_id, []).append(trial_id)

            self._push_log_replay_result(log, trial_id)

        elif op == JournalOperation.SET_TRIAL_PARAM:
            trial_id = log["trial_id"]

            if trial_id not in self._trials.keys():
                self._push_log_replay_result(log, KeyError)
                return

            if self._trials[trial_id].state.is_finished():
                self._push_log_replay_result(log, RuntimeError)
                return

            param_name = log["param_name"]
            param_value_external = log["param_value_external"]
            distribution = json_to_distribution(log["distribution"])

            self._trials[trial_id].params[param_name] = distribution.to_internal_repr(
                param_value_external
            )
            self._trials[trial_id].distributions[param_name] = distribution

            self._push_log_replay_result(log, None)

        elif op == JournalOperation.SET_TRIAL_STATE_VALUES:
            trial_id = log["trial_id"]

            if trial_id not in self._trials.keys():
                self._push_log_replay_result(log, KeyError)
                return

            if self._trials[trial_id].state.is_finished():
                self._push_log_replay_result(log, RuntimeError)
                return

            state = log["state"]
            values = log["values"]

            if state == self._trials[trial_id].state and state == TrialState.RUNNING:
                self._push_log_replay_result(log, False)
            else:
                self._trials[trial_id].state = state
                if values is not None:
                    self._trials[trial_id].values = values
                self._push_log_replay_result(log, True)

        elif op == JournalOperation.SET_TRIAL_INTERMEDIATE_VALUE:
            trial_id = log["trial_id"]

            if trial_id not in self._trials.keys():
                self._push_log_replay_result(log, KeyError)
                return

            if self._trials[trial_id].state.is_finished():
                self._push_log_replay_result(log, RuntimeError)
                return

            step = log["step"]
            intermediate_value = log["intermediate_value"]
            self._trials[trial_id].intermediate_values[step] = intermediate_value
            self._push_log_replay_result(log, None)

        else:
            raise RuntimeError("No corresponding log operation to op_code:{}".format(op))

    def _sync_with_backend(self) -> None:
        logs = self._backend.get_unread_logs(self._log_number_read)
        for log in logs:
            self._apply_log(log)
            self._log_number_read += 1

    # TODO(wattlebirdaz): Guarantee uniqueness.
    def _create_unique_study_name(self) -> str:
        DEFAULT_STUDY_NAME_PREFIX = "no-name-"
        return DEFAULT_STUDY_NAME_PREFIX + str(uuid.uuid4())

    # Basic study manipulation

    def create_new_study(self, study_name: Optional[str] = None) -> int:
        """Create a new study from a name.

        If no name is specified, the storage class generates a name.
        The returned study ID is unique among all current and deleted studies.

        Args:
            study_name:
                Name of the new study to create.

        Returns:
            ID of the created study.

        Raises:
            :exc:`optuna.exceptions.DuplicatedStudyError`:
                If a study with the same ``study_name`` already exists.
        """
        log = self._create_operation_log(JournalOperation.CREATE_STUDY)
        log["study_name"] = self._create_unique_study_name() if study_name is None else study_name
        self._buffer_log(log)
        self._flush_logs()

        with self._thread_lock:
            self._sync_with_backend()

            result = self._pop_log_replay_result(log)
            if isinstance(result, Exception):
                raise result
            else:
                assert isinstance(result, int)
                return result

    def delete_study(self, study_id: int) -> None:
        """Delete a study.

        Args:
            study_id:
                ID of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        log = self._create_operation_log(JournalOperation.DELETE_STUDY)
        log["study_id"] = study_id

        self._buffer_log(log)
        self._flush_logs()

        with self._thread_lock:
            self._sync_with_backend()

            result = self._pop_log_replay_result(log)
            if isinstance(result, Exception):
                raise result
            else:
                assert result is None
                return

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        """Register a user-defined attribute to a study.

        This method overwrites any existing attribute.

        Args:
            study_id:
                ID of the study.
            key:
                Attribute key.
            value:
                Attribute value. It should be JSON serializable.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        log = self._create_operation_log(JournalOperation.SET_STUDY_USER_ATTR)
        log["study_id"] = study_id
        log["user_attr"] = {key: value}

        self._buffer_log(log)
        self._flush_logs()

        with self._thread_lock:
            self._sync_with_backend()

            result = self._pop_log_replay_result(log)
            if isinstance(result, Exception):
                raise result
            else:
                assert result is None
                return

    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:
        """Register an optuna-internal attribute to a study.

        This method overwrites any existing attribute.

        Args:
            study_id:
                ID of the study.
            key:
                Attribute key.
            value:
                Attribute value. It should be JSON serializable.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        log = self._create_operation_log(JournalOperation.SET_STUDY_SYSTEM_ATTR)
        log["study_id"] = study_id
        log["system_attr"] = {key: value}

        self._buffer_log(log)
        self._flush_logs()

        with self._thread_lock:
            self._sync_with_backend()

            result = self._pop_log_replay_result(log)
            if isinstance(result, Exception):
                raise result
            else:
                assert result is None
                return

    def set_study_directions(self, study_id: int, directions: Sequence[StudyDirection]) -> None:
        """Register optimization problem directions to a study.

        Args:
            study_id:
                ID of the study.
            directions:
                A sequence of direction whose element is either
                :obj:`~optuna.study.StudyDirection.MAXIMIZE` or
                :obj:`~optuna.study.StudyDirection.MINIMIZE`.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
            :exc:`ValueError`:
                If the directions are already set and the each coordinate of passed ``directions``
                is the opposite direction or :obj:`~optuna.study.StudyDirection.NOT_SET`.
        """

        log = self._create_operation_log(JournalOperation.SET_STUDY_DIRECTIONS)
        log["study_id"] = study_id
        log["directions"] = directions

        self._buffer_log(log)
        self._flush_logs()

        with self._thread_lock:
            self._sync_with_backend()

            result = self._pop_log_replay_result(log)
            if isinstance(result, Exception):
                raise result
            else:
                assert result is None
                return

    # Basic study access

    def get_study_id_from_name(self, study_name: str) -> int:
        """Read the ID of a study.

        Args:
            study_name:
                Name of the study.

        Returns:
            ID of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_name`` exists.
        """
        with self._thread_lock:
            self._sync_with_backend()
            frozen_study = [fs for fs in self._studies.values() if fs.study_name == study_name]
            if len(frozen_study) == 0:
                raise KeyError
            assert len(frozen_study) == 1
            return frozen_study[0]._study_id

    def get_study_name_from_id(self, study_id: int) -> str:
        """Read the study name of a study.

        Args:
            study_id:
                ID of the study.

        Returns:
            Name of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        with self._thread_lock:
            self._sync_with_backend()
            if study_id not in self._studies.keys():
                raise KeyError
            else:
                return self._studies[study_id].study_name

    def get_study_directions(self, study_id: int) -> List[StudyDirection]:
        """Read whether a study maximizes or minimizes an objective.

        Args:
            study_id:
                ID of a study.

        Returns:
            Optimization directions list of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        with self._thread_lock:
            self._sync_with_backend()
            if study_id not in self._studies.keys():
                raise KeyError
            else:
                return self._studies[study_id].directions

    def get_study_user_attrs(self, study_id: int) -> Dict[str, Any]:
        """Read the user-defined attributes of a study.

        Args:
            study_id:
                ID of the study.

        Returns:
            Dictionary with the user attributes of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        with self._thread_lock:
            self._sync_with_backend()
            if study_id not in self._studies.keys():
                raise KeyError
            else:
                return self._studies[study_id].user_attrs

    def get_study_system_attrs(self, study_id: int) -> Dict[str, Any]:
        """Read the optuna-internal attributes of a study.

        Args:
            study_id:
                ID of the study.

        Returns:
            Dictionary with the optuna-internal attributes of the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        with self._thread_lock:
            self._sync_with_backend()
            if study_id not in self._studies.keys():
                raise KeyError
            else:
                return self._studies[study_id].system_attrs

    def get_all_studies(self) -> List[FrozenStudy]:
        """Read a list of :class:`~optuna.study.FrozenStudy` objects.

        Returns:
            A list of :class:`~optuna.study.FrozenStudy` objects.

        """
        with self._thread_lock:
            self._sync_with_backend()
            return list(self._studies.values())

    # Basic trial manipulation
    def create_new_trial(self, study_id: int, template_trial: Optional[FrozenTrial] = None) -> int:
        """Create and add a new trial to a study.

        The returned trial ID is unique among all current and deleted trials.

        Args:
            study_id:
                ID of the study.
            template_trial:
                Template :class:`~optuna.trial.FronzenTrial` with default user-attributes,
                system-attributes, intermediate-values, and a state.

        Returns:
            ID of the created trial.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        log = self._create_operation_log(JournalOperation.CREATE_TRIAL)
        log["study_id"] = study_id

        if template_trial is None:
            log["has_template_trial"] = False
        else:
            log["has_template_trial"] = True
            log["user_attrs"] = template_trial.user_attrs
            log["system_attrs"] = template_trial.system_attrs
            log["intermediate_values"] = template_trial.intermediate_values
            log["state"] = template_trial.state

        self._buffer_log(log)
        self._flush_logs()

        with self._thread_lock:
            self._sync_with_backend()
            result = self._pop_log_replay_result(log)
            if isinstance(result, Exception):
                raise result
            else:
                assert isinstance(result, int)
                return result

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
    ) -> None:
        """Set a parameter to a trial.

        Args:
            trial_id:
                ID of the trial.
            param_name:
                Name of the parameter.
            param_value_internal:
                Internal representation of the parameter value.
            distribution:
                Sampled distribution of the parameter.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        log = self._create_operation_log(JournalOperation.SET_TRIAL_PARAM)
        log["trial_id"] = trial_id
        log["param_name"] = param_name
        log["param_value_external"] = distribution.to_external_repr(param_value_internal)
        log["distribution"] = distribution_to_json(distribution)

        self._buffer_log(log)
        self._flush_logs()

        with self._thread_lock:
            self._sync_with_backend()

            result = self._pop_log_replay_result(log)
            if isinstance(result, Exception):
                raise result
            else:
                assert result is None
                return

    def get_trial_id_from_study_id_trial_number(self, study_id: int, trial_number: int) -> int:
        """Read the trial ID of a trial.

        Args:
            study_id:
                ID of the study.
            trial_number:
                Number of the trial.

        Returns:
            ID of the trial.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``study_id`` and ``trial_number`` exists.
        """
        with self._thread_lock:
            self._sync_with_backend()
            if len(self._study_id_to_trial_ids[study_id]) <= trial_number:
                raise KeyError
            return self._study_id_to_trial_ids[study_id][trial_number]

    def get_trial_number_from_id(self, trial_id: int) -> int:
        """Read the trial number of a trial.

        .. note::

            The trial number is only unique within a study, and is sequential.

        Args:
            trial_id:
                ID of the trial.

        Returns:
            Number of the trial.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        with self._thread_lock:
            self._sync_with_backend()
            trial_number = [
                trial.number for trial in self._trials.values() if trial._trial_id == trial_id
            ]
            if len(trial_number) != 1:
                raise KeyError
            else:
                return trial_number[0]

    def get_trial_param(self, trial_id: int, param_name: str) -> float:
        """Read the parameter of a trial.

        Args:
            trial_id:
                ID of the trial.
            param_name:
                Name of the parameter.

        Returns:
            Internal representation of the parameter.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
                If no such parameter exists.
        """
        with self._thread_lock:
            self._sync_with_backend()
            frozen_trial = [
                trial for trial in self._trials.values() if trial._trial_id == trial_id
            ]
            if len(frozen_trial) != 1 or param_name not in frozen_trial[0].distributions.keys():
                raise KeyError
            return (
                frozen_trial[0]
                .distributions[param_name]
                .to_internal_repr(frozen_trial[0].params[param_name])
            )

    def set_trial_state_values(
        self, trial_id: int, state: TrialState, values: Optional[Sequence[float]] = None
    ) -> bool:
        """Update the state and values of a trial.

        Set return values of an objective function to values argument.
        If values argument is not :obj:`None`, this method overwrites any existing trial values.

        Args:
            trial_id:
                ID of the trial.
            state:
                New state of the trial.
            values:
                Values of the objective function.

        Returns:
            :obj:`True` if the state is successfully updated.
            :obj:`False` if the state is kept the same.
            The latter happens when this method tries to update the state of
            :obj:`~optuna.trial.TrialState.RUNNING` trial to
            :obj:`~optuna.trial.TrialState.RUNNING`.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        log = self._create_operation_log(JournalOperation.SET_TRIAL_STATE_VALUES)
        log["trial_id"] = trial_id
        log["state"] = state
        log["values"] = values

        self._buffer_log(log)
        self._flush_logs()

        with self._thread_lock:
            self._sync_with_backend()

            result = self._pop_log_replay_result(log)
            if isinstance(result, Exception):
                raise result
            else:
                assert isinstance(result, bool)
                return result

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        """Report an intermediate value of an objective function.

        This method overwrites any existing intermediate value associated with the given step.

        Args:
            trial_id:
                ID of the trial.
            step:
                Step of the trial (e.g., the epoch when training a neural network).
            intermediate_value:
                Intermediate value corresponding to the step.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        log = self._create_operation_log(JournalOperation.SET_TRIAL_INTERMEDIATE_VALUE)
        log["trial_id"] = trial_id
        log["step"] = step
        log["intermediate_value"] = intermediate_value

        self._buffer_log(log)
        self._flush_logs()

        with self._thread_lock:
            self._sync_with_backend()
            result = self._pop_log_replay_result(log)
            if isinstance(result, Exception):
                raise result
            else:
                return result

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        """Set a user-defined attribute to a trial.

        This method overwrites any existing attribute.

        Args:
            trial_id:
                ID of the trial.
            key:
                Attribute key.
            value:
                Attribute value. It should be JSON serializable.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """

        raise NotImplementedError

    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:
        """Set an optuna-internal attribute to a trial.

        This method overwrites any existing attribute.

        Args:
            trial_id:
                ID of the trial.
            key:
                Attribute key.
            value:
                Attribute value. It should be JSON serializable.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        raise NotImplementedError

    # Basic trial access

    def get_trial(self, trial_id: int) -> FrozenTrial:
        """Read a trial.

        Args:
            trial_id:
                ID of the trial.

        Returns:
            Trial with a matching trial ID.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        raise NotImplementedError

    def get_all_trials(
        self,
        study_id: int,
        deepcopy: bool = True,
        states: Optional[Container[TrialState]] = None,
    ) -> List[FrozenTrial]:
        """Read all trials in a study.

        Args:
            study_id:
                ID of the study.
            deepcopy:
                Whether to copy the list of trials before returning.
                Set to :obj:`True` if you intend to update the list or elements of the list.
            states:
                Trial states to filter on. If :obj:`None`, include all states.

        Returns:
            List of trials in the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        raise NotImplementedError

    def get_n_trials(
        self,
        study_id: int,
        state: Optional[Union[Tuple[TrialState, ...], TrialState]] = None,
    ) -> int:
        """Count the number of trials in a study.

        Args:
            study_id:
                ID of the study.
            state:
                Trial states to filter on. If :obj:`None`, include all states.

        Returns:
            Number of trials in the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
        """
        # TODO(hvy): Align the name and the behavior or the `state` parameter with
        # `get_all_trials`'s `states`.
        if isinstance(state, TrialState):
            state = (state,)
        return len(self.get_all_trials(study_id, deepcopy=False, states=state))

    def get_best_trial(self, study_id: int) -> FrozenTrial:
        """Return the trial with the best value in a study.

        This method is valid only during single-objective optimization.

        Args:
            study_id:
                ID of the study.

        Returns:
            The trial with the best objective value among all finished trials in the study.

        Raises:
            :exc:`KeyError`:
                If no study with the matching ``study_id`` exists.
            :exc:`RuntimeError`:
                If the study has more than one direction.
            :exc:`ValueError`:
                If no trials have been completed.
        """
        all_trials = self.get_all_trials(study_id, deepcopy=False)
        all_trials = [t for t in all_trials if t.state is TrialState.COMPLETE]

        if len(all_trials) == 0:
            raise ValueError("No trials are completed yet.")

        directions = self.get_study_directions(study_id)
        if len(directions) > 1:
            raise RuntimeError(
                "Best trial can be obtained only for single-objective optimization."
            )
        direction = directions[0]

        if direction == StudyDirection.MAXIMIZE:
            best_trial = max(all_trials, key=lambda t: cast(float, t.value))
        else:
            best_trial = min(all_trials, key=lambda t: cast(float, t.value))

        return best_trial

    def get_trial_params(self, trial_id: int) -> Dict[str, Any]:
        """Read the parameter dictionary of a trial.

        Args:
            trial_id:
                ID of the trial.

        Returns:
            Dictionary of a parameters. Keys are parameter names and values are internal
            representations of the parameter values.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        return self.get_trial(trial_id).params

    def get_trial_user_attrs(self, trial_id: int) -> Dict[str, Any]:
        """Read the user-defined attributes of a trial.

        Args:
            trial_id:
                ID of the trial.

        Returns:
            Dictionary with the user-defined attributes of the trial.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        return self.get_trial(trial_id).user_attrs

    def get_trial_system_attrs(self, trial_id: int) -> Dict[str, Any]:
        """Read the optuna-internal attributes of a trial.

        Args:
            trial_id:
                ID of the trial.

        Returns:
            Dictionary with the optuna-internal attributes of the trial.

        Raises:
            :exc:`KeyError`:
                If no trial with the matching ``trial_id`` exists.
        """
        return self.get_trial(trial_id).system_attrs

    def remove_session(self) -> None:
        """Clean up all connections to a database."""
        pass

    def check_trial_is_updatable(self, trial_id: int, trial_state: TrialState) -> None:
        """Check whether a trial state is updatable.

        Args:
            trial_id:
                ID of the trial.
                Only used for an error message.
            trial_state:
                Trial state to check.

        Raises:
            :exc:`RuntimeError`:
                If the trial is already finished.
        """
        if trial_state.is_finished():
            trial = self.get_trial(trial_id)
            raise RuntimeError(
                "Trial#{} has already finished and can not be updated.".format(trial.number)
            )
