import copy
from datetime import datetime
import pickle
import redis
import threading

from optuna import distributions, exceptions  # NOQA
from optuna.storages import base
from optuna.storages.base import DEFAULT_STUDY_NAME_PREFIX
from optuna import structs
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA


class RedisStorage(base.BaseStorage):
    """Storage class for Redis backend.

    Note that library users can instantiate this class, but the attributes
    provided by this class are not supposed to be directly accessed by them.

    Example:

        We create an :class:`~optuna.storages.redis_storage.RedisStorage` instance using
        the given redis database URL

        .. code::

            >>> import optuna
            >>>
            >>> def objective(trial):
            >>>     ...
            >>>
            >>> storage = optuna.storages.redis_storage.RedisStorage(
            >>>     url='redis://passwd@localhost:port/db',
            >>>     testing_environment=False
            >>> )
            >>>
            >>> study = optuna.create_study(storage=storage)
            >>> study.optimize(objective)

    Args:
        url: URL of the redis storage. password and db are optional
                   (ie: redis://localhost:6379)
        testing_environment: Defaults to False

    """

    def __init__(self, url):
        # type: (str) -> None

        self.redis = redis.Redis.from_url(url)
        self._lock = threading.RLock()

    @staticmethod
    def init_with_flush(url):
        # type: (str) -> base.BaseStorage

        storage = RedisStorage(url)
        storage.redis.flushdb()
        return storage

    def __getstate__(self):
        # type: () -> Dict[Any, Any]

        state = self.__dict__.copy()
        del state['_lock']
        return state

    def __setstate__(self, state):
        # type: (Dict[Any, Any]) -> None

        self.__dict__.update(state)
        self._lock = threading.RLock()

    def create_new_study(self, study_name=None):
        # type: (Optional[str]) -> int

        key_study_name = "study_name:{}:study_id".format(study_name)
        if study_name is not None and self.redis.exists(key_study_name):
            raise exceptions.DuplicatedStudyError

        if not self.redis.exists("study_counter"):
            # we need the counter to start with 0
            self.redis.set("study_counter", -1)
        study_id = self.redis.incr("study_counter", 1)
        # we need the trial_number counter to start with 0
        self.redis.set("study_id:{:010d}:trial_number".format(study_id), -1)

        if study_name is None:
            study_name = '{}{:010d}'.format(DEFAULT_STUDY_NAME_PREFIX, study_id)

        self.redis.set(key_study_name, pickle.dumps(study_id))
        self.redis.set("study_id:{:010d}:study_name".format(study_id), pickle.dumps(study_name))
        self.redis.set("study_id:{:010d}:direction".format(study_id),
                       pickle.dumps(structs.StudyDirection.NOT_SET))

        study_summary = structs.StudySummary(
            study_name=study_name,
            direction=structs.StudyDirection.NOT_SET,
            best_trial=None,
            user_attrs={},
            system_attrs={},
            n_trials=0,
            datetime_start=datetime.now(),
            study_id=study_id
        )
        self.redis.rpush("study_list", pickle.dumps(study_id))
        self._set_study_summary(study_id, study_summary)

        return study_id

    def delete_study(self, study_id):
        # type: (int) -> None

        self._check_study_id(study_id)

        with self._lock:
            # Sumaries
            self._del_study_summary(study_id)
            self.redis.lrem("study_list", 0, pickle.dumps(study_id))
            # Trials
            trial_ids = self._get_study_trials(study_id)
            for trial_id in trial_ids:
                self._del_trial(trial_id)
            self.redis.delete("study_id:{:010d}:trial_list".format(study_id))
            self.redis.delete("study_id:{:010d}:trial_number".format(study_id))
            # Study
            study_name = self.get_study_name_from_id(study_id)
            self.redis.delete("study_name:{}:study_id".format(study_name))
            self.redis.delete("study_id:{:010d}:study_name".format(study_id))
            self.redis.delete("study_id:{:010d}:direction".format(study_id))
            self.redis.delete("study_id:{:010d}:best_trial_id".format(study_id))
            self.redis.delete("study_id:{:010d}:params_distribution".format(study_id))

    @staticmethod
    def _key_study_summary(study_id):
        # type: (int) -> str

        return "study_id:{:010d}:study_summary".format(study_id)

    def _set_study_summary(self, study_id, study_summary):
        # type: (int, structs.StudySummary) -> None

        self.redis.set(self._key_study_summary(study_id), pickle.dumps(study_summary))

    def _get_study_summary(self, study_id):
        # type: (int) -> structs.StudySummary

        return pickle.loads(self.redis.get(self._key_study_summary(study_id)))

    def _del_study_summary(self, study_id):
        # type: (int) -> None

        self.redis.delete(self._key_study_summary(study_id))

    def set_study_direction(self, study_id, direction):
        # type: (int, structs.StudyDirection) -> None

        if self.redis.exists("study_id:{:010d}:direction".format(study_id)):
            current_direction = pickle.loads(
                self.redis.get("study_id:{:010d}:direction".format(study_id)))
            if current_direction != structs.StudyDirection.NOT_SET and \
               current_direction != direction:
                raise ValueError('Cannot overwrite study direction from {} to {}.'.format(
                    current_direction, direction))

        self.redis.set("study_id:{:010d}:direction".format(study_id), pickle.dumps(direction))
        study_summary = self._get_study_summary(study_id)
        study_summary.direction = direction
        self._set_study_summary(study_id, study_summary)

    def set_study_user_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        with self._lock:
            study_summary = self._get_study_summary(study_id)
            study_summary.user_attrs[key] = value
            self._set_study_summary(study_id, study_summary)

    def set_study_system_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        with self._lock:
            study_summary = self._get_study_summary(study_id)
            study_summary.system_attrs[key] = value
            self._set_study_summary(study_id, study_summary)

    def get_study_id_from_name(self, study_name):
        # type: (str) -> int

        if not self.redis.exists("study_name:{}:study_id".format(study_name)):
            raise ValueError("No such study {}.".format(study_name))

        study_id = pickle.loads(self.redis.get("study_name:{}:study_id".format(study_name)))

        return study_id

    def get_study_id_from_trial_id(self, trial_id):
        # type: (int) -> int

        study_id = pickle.loads(self.redis.get("trial_id:{:010d}:study_id".format(trial_id)))
        return study_id

    def get_study_name_from_id(self, study_id):
        # type: (int) -> str

        self._check_study_id(study_id)
        study_name = pickle.loads(self.redis.get("study_id:{:010d}:study_name".format(study_id)))
        return study_name

    def get_study_direction(self, study_id):
        # type: (int) -> structs.StudyDirection

        return pickle.loads(self.redis.get("study_id:{:010d}:direction".format(study_id)))

    def get_study_user_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        with self._lock:
            study_summary = self._get_study_summary(study_id)
            return copy.deepcopy(study_summary.user_attrs)

    def get_study_system_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        with self._lock:
            study_summary = self._get_study_summary(study_id)
            return copy.deepcopy(study_summary.system_attrs)

    @staticmethod
    def _key_study_param_distribution(study_id):
        # type: (int) -> str

        return "study_id:{:010d}:params_distribution".format(study_id)

    def _get_study_param_distribution(self, study_id):
        # type: (int) -> Dict

        if self.redis.exists(self._key_study_param_distribution(study_id)):
            return pickle.loads(self.redis.get(self._key_study_param_distribution(study_id)))
        else:
            return {}

    def _set_study_param_distribution(self, study_id, param_distribution):
        # type: (int, Dict) -> None

        self.redis.set(self._key_study_param_distribution(study_id),
                       pickle.dumps(param_distribution))

    def get_all_study_summaries(self):
        # type: () -> List[structs.StudySummary]

        with self._lock:
            study_summaries = []
            study_ids = [pickle.loads(sid) for sid in self.redis.lrange("study_list", 0, -1)]
            for study_id in study_ids:
                study_summary = self._get_study_summary(study_id)
                study_summaries.append(study_summary)

        return study_summaries

    def create_new_trial(self, study_id, template_trial=None):
        # type: (int, Optional[structs.FrozenTrial]) -> int

        self._check_study_id(study_id)

        if template_trial is None:
            trial = self._create_running_trial()
        else:
            trial = copy.deepcopy(template_trial)

        if not self.redis.exists("trial_counter"):
            self.redis.set("trial_counter", -1)

        trial_id = self.redis.incr("trial_counter", 1)
        trial_number = self.redis.incr("study_id:{:010d}:trial_number".format(study_id))
        trial.system_attrs['_number'] = trial_number
        trial.number = trial_number
        trial._trial_id = trial_id

        self.redis.set(self._key_trial(trial_id), pickle.dumps(trial))
        self.redis.set("trial_id:{:010d}:study_id".format(trial_id), pickle.dumps(study_id))
        self.redis.rpush("study_id:{:010d}:trial_list".format(study_id), trial_id)

        study_summary = self._get_study_summary(study_id)
        study_summary.n_trials = len(self._get_study_trials(study_id))
        min_datetime_start = min([t.datetime_start for t in self.get_all_trials(study_id)])
        study_summary.datetime_start = min_datetime_start
        self._set_study_summary(study_id, study_summary)

        return trial_id

    @staticmethod
    def _create_running_trial():
        # type: () -> structs.FrozenTrial

        return structs.FrozenTrial(
            trial_id=-1,  # dummy value.
            number=-1,  # dummy value.
            state=structs.TrialState.RUNNING,
            params={},
            distributions={},
            user_attrs={},
            system_attrs={},
            value=None,
            intermediate_values={},
            datetime_start=datetime.now(),
            datetime_complete=None)

    def set_trial_state(self, trial_id, state):
        # type: (int, structs.TrialState) -> bool

        with self._lock:
            trial = self.get_trial(trial_id)
            self.check_trial_is_updatable(trial_id, trial.state)

            if state == structs.TrialState.RUNNING and trial.state != structs.TrialState.WAITING:
                return False

            trial.state = state
            if state.is_finished():
                trial.datetime_complete = datetime.now()
                self.redis.set(self._key_trial(trial_id), pickle.dumps(trial))
                self._update_cache(trial_id)
            else:
                self.redis.set(self._key_trial(trial_id), pickle.dumps(trial))

        return True

    def set_trial_param(self, trial_id, param_name, param_value_internal, distribution):
        # type: (int, str, float, distributions.BaseDistribution) -> bool

        with self._lock:
            self.check_trial_is_updatable(trial_id, self.get_trial(trial_id).state)

            # Check param distribution compatibility with previous trial(s).
            study_id = self.get_study_id_from_trial_id(trial_id)
            param_distribution = self._get_study_param_distribution(study_id)
            if param_name in param_distribution:
                distributions.check_distribution_compatibility(param_distribution[param_name],
                                                               distribution)

            trial = self.get_trial(trial_id)
            # Check param has not been set; otherwise, return False.
            if param_name in trial.params:
                return False

            # Set study param distribution.
            param_distribution[param_name] = distribution
            self._set_study_param_distribution(study_id, param_distribution)

            # Set params.
            trial.params[param_name] = distribution.to_external_repr(param_value_internal)
            trial.distributions[param_name] = distribution
            self.redis.set(self._key_trial(trial_id), pickle.dumps(trial))

            return True

    def get_trial_number_from_id(self, trial_id):
        # type: (int) -> int

        return self.get_trial(trial_id).number

    @staticmethod
    def _key_best_trial(study_id):
        # type: (int) -> str

        return "study_id:{:010d}:best_trial_id".format(study_id)

    def get_best_trial(self, study_id):
        # type: (int) -> structs.FrozenTrial

        if not self.redis.exists(self._key_best_trial(study_id)):
            all_trials = self.get_all_trials(study_id, deepcopy=False)
            all_trials = [t for t in all_trials if t.state is structs.TrialState.COMPLETE]

            if len(all_trials) == 0:
                raise ValueError('No trials are completed yet.')

            if self.get_study_direction(study_id) == structs.StudyDirection.MAXIMIZE:
                best_trial = max(all_trials, key=lambda t: t.value)
            else:
                best_trial = min(all_trials, key=lambda t: t.value)

            self._set_best_trial(study_id, best_trial.number)
        else:
            best_trial_id = pickle.loads(self.redis.get(self._key_best_trial(study_id)))
            best_trial = self.get_trial(best_trial_id)

        return best_trial

    def _set_best_trial(self, study_id, trial_id):
        # type: (int, int) -> None

        self.redis.set(self._key_best_trial(study_id), pickle.dumps(trial_id))

        study_summary = self._get_study_summary(study_id)
        study_summary.best_trial = self.get_trial(trial_id)
        self._set_study_summary(study_id, study_summary)

    def get_trial_param(self, trial_id, param_name):
        # type: (int, str) -> float

        distribution = self.get_trial(trial_id).distributions[param_name]
        return distribution.to_internal_repr(self.get_trial(trial_id).params[param_name])

    def set_trial_value(self, trial_id, value):
        # type: (int, float) -> None

        with self._lock:
            trial = self.get_trial(trial_id)
            self.check_trial_is_updatable(trial_id, trial.state)

            trial.value = value
            self.redis.set(self._key_trial(trial_id), pickle.dumps(trial))

    def _update_cache(self, trial_id):
        # type: (int) -> None

        trial = self.get_trial(trial_id)
        if trial.state != structs.TrialState.COMPLETE:
            return
        study_id = self.get_study_id_from_trial_id(trial_id)
        if not self.redis.exists("study_id:{:010d}:best_trial_id".format(study_id)):
            self._set_best_trial(study_id, trial_id)
            return

        best_value = float(self.get_best_trial(study_id).value)
        new_value = float(trial.value)

        # Complete trials do not have `None` values.
        assert new_value is not None

        if self.get_study_direction(study_id) == structs.StudyDirection.MAXIMIZE:
            if new_value > best_value:
                self._set_best_trial(study_id, trial_id)
        else:
            if new_value < best_value:
                self._set_best_trial(study_id, trial_id)

        return

    def set_trial_intermediate_value(self, trial_id, step, intermediate_value):
        # type: (int, int, float) -> bool

        with self._lock:
            self.check_trial_is_updatable(trial_id, self.get_trial(trial_id).state)

            frozen_trial = self.get_trial(trial_id)
            if step in frozen_trial.intermediate_values:
                return False

            frozen_trial.intermediate_values[step] = intermediate_value
            self._set_trial(trial_id, frozen_trial)

            return True

    def set_trial_user_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        with self._lock:
            trial = self.get_trial(trial_id)
            self.check_trial_is_updatable(trial_id, trial.state)
            trial.user_attrs[key] = value
            self._set_trial(trial_id, trial)

    def set_trial_system_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        with self._lock:
            trial = self.get_trial(trial_id)
            self.check_trial_is_updatable(trial_id, trial.state)
            trial.system_attrs[key] = value
            self._set_trial(trial_id, trial)

    @staticmethod
    def _key_trial(trial_id):
        # type: (int) -> str

        return "trial_id:{:010d}:frozentrial".format(trial_id)

    def get_trial(self, trial_id):
        # type: (int) -> structs.FrozenTrial

        return pickle.loads(self.redis.get(self._key_trial(trial_id)))

    def _set_trial(self, trial_id, trial):
        # type: (int, structs.FrozenTrial) -> None

        self.redis.set(self._key_trial(trial_id), pickle.dumps(trial))

    def _del_trial(self, trial_id):
        # type: (int) -> None

        self.redis.delete(self._key_trial(trial_id))
        self.redis.delete("trial_id:{:010d}:study_id".format(trial_id))

    def _get_study_trials(self, study_id):
        # type: (int) -> List[int]

        self._check_study_id(study_id)

        study_trial_list_key = "study_id:{:010d}:trial_list".format(study_id)
        return [int(tid) for tid in self.redis.lrange(study_trial_list_key, 0, -1)]

    def get_all_trials(self, study_id, deepcopy=True):
        # type: (int, bool) -> List[structs.FrozenTrial]

        self._check_study_id(study_id)

        with self._lock:
            trials = []
            trial_ids = self._get_study_trials(study_id)
            for trial_id in trial_ids:
                frozen_trial = self.get_trial(trial_id)
                trials.append(frozen_trial)

        if deepcopy:
            return copy.deepcopy(trials)
        else:
            return trials

    def get_n_trials(self, study_id, state=None):
        # type: (int, Optional[structs.TrialState]) -> int

        self._check_study_id(study_id)
        if state is None:
            return len(self.get_all_trials(study_id))

        return len([t for t in self.get_all_trials(study_id) if t.state == state])

    def _check_study_id(self, study_id):
        # type: (int) -> None

        if not self.redis.exists("study_id:{:010d}:study_name".format(study_id)):
            raise ValueError('study_id {} does not exist.'.format(study_id))
