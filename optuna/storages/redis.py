import copy
from datetime import datetime
import pickle

from optuna._experimental import experimental
from optuna import distributions
from optuna import exceptions
from optuna.storages import base
from optuna.storages.base import DEFAULT_STUDY_NAME_PREFIX
from optuna.study import StudyDirection
from optuna.study import StudySummary
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna import type_checking

try:
    import redis

    _available = True
except ImportError as e:
    _import_error = e
    _available = False

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA


@experimental("1.4.0")
class RedisStorage(base.BaseStorage):
    """Storage class for Redis backend.

    Note that library users can instantiate this class, but the attributes
    provided by this class are not supposed to be directly accessed by them.

    Example:

        We create an :class:`~optuna.storages.redis.RedisStorage` instance using
        the given redis database URL.

        .. code::

            >>> import optuna
            >>>
            >>> def objective(trial):
            >>>     ...
            >>>
            >>> storage = optuna.storages.redis.RedisStorage(
            >>>     url='redis://passwd@localhost:port/db',
            >>> )
            >>>
            >>> study = optuna.create_study(storage=storage)
            >>> study.optimize(objective)

    Args:
        url: URL of the redis storage, password and db are optional. (ie: redis://localhost:6379)

    .. note::
        If you use plan to use Redis as a storage mechanism for optuna,
        make sure Redis in installed and running.
        Please execute ``$ pip install -U redis`` to install redis python library.
    """

    def __init__(self, url):
        # type: (str) -> None

        if not _available:
            raise ImportError(
                "Redis in not available. Please install redis to use this feature. "
                "Redis can be installed by executing `$ pip install -U redis`. "
                "For further information, please refer to the installation guide of redis. "
                "(The actual import error is as follows: " + str(_import_error) + ")"
            )

        self._redis = redis.Redis.from_url(url)

    def create_new_study(self, study_name=None):
        # type: (Optional[str]) -> int

        if study_name is not None and self._redis.exists(self._key_study_name(study_name)):
            raise exceptions.DuplicatedStudyError

        if not self._redis.exists("study_counter"):
            # We need the counter to start with 0.
            self._redis.set("study_counter", -1)
        study_id = self._redis.incr("study_counter", 1)
        # We need the trial_number counter to start with 0.
        self._redis.set("study_id:{:010d}:trial_number".format(study_id), -1)

        if study_name is None:
            study_name = "{}{:010d}".format(DEFAULT_STUDY_NAME_PREFIX, study_id)

        with self._redis.pipeline() as pipe:
            pipe.multi()
            pipe.set(self._key_study_name(study_name), pickle.dumps(study_id))
            pipe.set("study_id:{:010d}:study_name".format(study_id), pickle.dumps(study_name))
            pipe.set(
                "study_id:{:010d}:direction".format(study_id),
                pickle.dumps(StudyDirection.NOT_SET),
            )

            study_summary = StudySummary(
                study_name=study_name,
                direction=StudyDirection.NOT_SET,
                best_trial=None,
                user_attrs={},
                system_attrs={},
                n_trials=0,
                datetime_start=None,
                study_id=study_id,
            )
            pipe.rpush("study_list", pickle.dumps(study_id))
            pipe.set(self._key_study_summary(study_id), pickle.dumps(study_summary))
            pipe.execute()

        return study_id

    def delete_study(self, study_id):
        # type: (int) -> None

        self._check_study_id(study_id)

        with self._redis.pipeline() as pipe:
            pipe.multi()
            # Sumaries
            pipe.delete(self._key_study_summary(study_id))
            pipe.lrem("study_list", 0, pickle.dumps(study_id))
            # Trials
            trial_ids = self._get_study_trials(study_id)
            for trial_id in trial_ids:
                pipe.delete("trial_id:{:010d}:frozentrial".format(trial_id))
                pipe.delete("trial_id:{:010d}:study_id".format(trial_id))
            pipe.delete("study_id:{:010d}:trial_list".format(study_id))
            pipe.delete("study_id:{:010d}:trial_number".format(study_id))
            # Study
            study_name = self.get_study_name_from_id(study_id)
            pipe.delete("study_name:{}:study_id".format(study_name))
            pipe.delete("study_id:{:010d}:study_name".format(study_id))
            pipe.delete("study_id:{:010d}:direction".format(study_id))
            pipe.delete("study_id:{:010d}:best_trial_id".format(study_id))
            pipe.delete("study_id:{:010d}:params_distribution".format(study_id))
            pipe.execute()

    @staticmethod
    def _key_study_name(study_name):
        # type: (str) -> str

        return "study_name:{}:study_id".format(study_name)

    @staticmethod
    def _key_study_summary(study_id):
        # type: (int) -> str

        return "study_id:{:010d}:study_summary".format(study_id)

    def _set_study_summary(self, study_id, study_summary):
        # type: (int, StudySummary) -> None

        self._redis.set(self._key_study_summary(study_id), pickle.dumps(study_summary))

    def _get_study_summary(self, study_id):
        # type: (int) -> StudySummary

        summary_pkl = self._redis.get(self._key_study_summary(study_id))
        assert summary_pkl is not None
        return pickle.loads(summary_pkl)

    def _del_study_summary(self, study_id):
        # type: (int) -> None

        self._redis.delete(self._key_study_summary(study_id))

    @staticmethod
    def _key_study_direction(study_id):
        # type: (int) -> str

        return "study_id:{:010d}:direction".format(study_id)

    def set_study_direction(self, study_id, direction):
        # type: (int, StudyDirection) -> None

        self._check_study_id(study_id)

        if self._redis.exists(self._key_study_direction(study_id)):
            direction_pkl = self._redis.get(self._key_study_direction(study_id))
            assert direction_pkl is not None
            current_direction = pickle.loads(direction_pkl)
            if current_direction != StudyDirection.NOT_SET and current_direction != direction:
                raise ValueError(
                    "Cannot overwrite study direction from {} to {}.".format(
                        current_direction, direction
                    )
                )

        with self._redis.pipeline() as pipe:
            pipe.multi()
            pipe.set(self._key_study_direction(study_id), pickle.dumps(direction))
            study_summary = self._get_study_summary(study_id)
            study_summary.direction = direction
            pipe.set(self._key_study_summary(study_id), pickle.dumps(study_summary))
            pipe.execute()

    def set_study_user_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        self._check_study_id(study_id)

        study_summary = self._get_study_summary(study_id)
        study_summary.user_attrs[key] = value
        self._set_study_summary(study_id, study_summary)

    def set_study_system_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        self._check_study_id(study_id)

        study_summary = self._get_study_summary(study_id)
        study_summary.system_attrs[key] = value
        self._set_study_summary(study_id, study_summary)

    def get_study_id_from_name(self, study_name):
        # type: (str) -> int

        if not self._redis.exists(self._key_study_name(study_name)):
            raise KeyError("No such study {}.".format(study_name))
        study_id_pkl = self._redis.get(self._key_study_name(study_name))
        assert study_id_pkl is not None
        return pickle.loads(study_id_pkl)

    def get_study_id_from_trial_id(self, trial_id):
        # type: (int) -> int

        study_id_pkl = self._redis.get("trial_id:{:010d}:study_id".format(trial_id))
        if study_id_pkl is None:
            raise KeyError("No such trial: {}.".format(trial_id))
        return pickle.loads(study_id_pkl)

    def get_study_name_from_id(self, study_id):
        # type: (int) -> str

        self._check_study_id(study_id)

        study_name_pkl = self._redis.get("study_id:{:010d}:study_name".format(study_id))
        if study_name_pkl is None:
            raise KeyError("No such study: {}.".format(study_id))
        return pickle.loads(study_name_pkl)

    def get_study_direction(self, study_id):
        # type: (int) -> StudyDirection

        direction_pkl = self._redis.get("study_id:{:010d}:direction".format(study_id))
        if direction_pkl is None:
            raise KeyError("No such study: {}.".format(study_id))
        return pickle.loads(direction_pkl)

    def get_study_user_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        self._check_study_id(study_id)

        study_summary = self._get_study_summary(study_id)
        return copy.deepcopy(study_summary.user_attrs)

    def get_study_system_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        self._check_study_id(study_id)

        study_summary = self._get_study_summary(study_id)
        return copy.deepcopy(study_summary.system_attrs)

    @staticmethod
    def _key_study_param_distribution(study_id):
        # type: (int) -> str

        return "study_id:{:010d}:params_distribution".format(study_id)

    def _get_study_param_distribution(self, study_id):
        # type: (int) -> Dict

        if self._redis.exists(self._key_study_param_distribution(study_id)):
            param_distribution_pkl = self._redis.get(self._key_study_param_distribution(study_id))
            assert param_distribution_pkl is not None
            return pickle.loads(param_distribution_pkl)
        else:
            return {}

    def _set_study_param_distribution(self, study_id, param_distribution):
        # type: (int, Dict) -> None

        self._redis.set(
            self._key_study_param_distribution(study_id), pickle.dumps(param_distribution)
        )

    def get_all_study_summaries(self):
        # type: () -> List[StudySummary]

        study_summaries = []
        study_ids = [pickle.loads(sid) for sid in self._redis.lrange("study_list", 0, -1)]
        for study_id in study_ids:
            study_summary = self._get_study_summary(study_id)
            study_summaries.append(study_summary)

        return study_summaries

    def create_new_trial(self, study_id, template_trial=None):
        # type: (int, Optional[FrozenTrial]) -> int

        self._check_study_id(study_id)

        if template_trial is None:
            trial = self._create_running_trial()
        else:
            trial = copy.deepcopy(template_trial)

        if not self._redis.exists("trial_counter"):
            self._redis.set("trial_counter", -1)

        trial_id = self._redis.incr("trial_counter", 1)
        trial_number = self._redis.incr("study_id:{:010d}:trial_number".format(study_id))
        trial.number = trial_number
        trial._trial_id = trial_id

        with self._redis.pipeline() as pipe:
            pipe.multi()
            pipe.set(self._key_trial(trial_id), pickle.dumps(trial))
            pipe.set("trial_id:{:010d}:study_id".format(trial_id), pickle.dumps(study_id))
            pipe.rpush("study_id:{:010d}:trial_list".format(study_id), trial_id)
            pipe.execute()

            pipe.multi()
            study_summary = self._get_study_summary(study_id)
            study_summary.n_trials = len(self._get_study_trials(study_id))
            min_datetime_start = min([t.datetime_start for t in self.get_all_trials(study_id)])
            study_summary.datetime_start = min_datetime_start
            pipe.set(self._key_study_summary(study_id), pickle.dumps(study_summary))
            pipe.execute()

        if trial.state.is_finished():
            self._update_cache(trial_id)

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

        self._check_trial_id(trial_id)
        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)

        if state == TrialState.RUNNING and trial.state != TrialState.WAITING:
            return False

        trial.state = state
        if state.is_finished():
            trial.datetime_complete = datetime.now()
            self._redis.set(self._key_trial(trial_id), pickle.dumps(trial))
            self._update_cache(trial_id)
        else:
            self._redis.set(self._key_trial(trial_id), pickle.dumps(trial))

        return True

    def set_trial_param(self, trial_id, param_name, param_value_internal, distribution):
        # type: (int, str, float, distributions.BaseDistribution) -> bool

        self._check_trial_id(trial_id)
        self.check_trial_is_updatable(trial_id, self.get_trial(trial_id).state)

        # Check param distribution compatibility with previous trial(s).
        study_id = self.get_study_id_from_trial_id(trial_id)
        param_distribution = self._get_study_param_distribution(study_id)
        if param_name in param_distribution:
            distributions.check_distribution_compatibility(
                param_distribution[param_name], distribution
            )

        trial = self.get_trial(trial_id)
        # Check param has not been set; otherwise, return False.
        if param_name in trial.params:
            return False

        with self._redis.pipeline() as pipe:
            pipe.multi()
            # Set study param distribution.
            param_distribution[param_name] = distribution
            pipe.set(
                self._key_study_param_distribution(study_id), pickle.dumps(param_distribution)
            )

            # Set params.
            trial.params[param_name] = distribution.to_external_repr(param_value_internal)
            trial.distributions[param_name] = distribution
            pipe.set(self._key_trial(trial_id), pickle.dumps(trial))
            pipe.execute()

        return True

    def get_trial_number_from_id(self, trial_id):
        # type: (int) -> int

        return self.get_trial(trial_id).number

    @staticmethod
    def _key_best_trial(study_id):
        # type: (int) -> str

        return "study_id:{:010d}:best_trial_id".format(study_id)

    def get_best_trial(self, study_id):
        # type: (int) -> FrozenTrial

        if not self._redis.exists(self._key_best_trial(study_id)):
            all_trials = self.get_all_trials(study_id, deepcopy=False)
            all_trials = [t for t in all_trials if t.state is TrialState.COMPLETE]

            if len(all_trials) == 0:
                raise ValueError("No trials are completed yet.")

            if self.get_study_direction(study_id) == StudyDirection.MAXIMIZE:
                best_trial = max(all_trials, key=lambda t: t.value)
            else:
                best_trial = min(all_trials, key=lambda t: t.value)

            self._set_best_trial(study_id, best_trial.number)
        else:
            best_trial_id_pkl = self._redis.get(self._key_best_trial(study_id))
            assert best_trial_id_pkl is not None
            best_trial_id = pickle.loads(best_trial_id_pkl)
            best_trial = self.get_trial(best_trial_id)

        return best_trial

    def _set_best_trial(self, study_id, trial_id):
        # type: (int, int) -> None

        with self._redis.pipeline() as pipe:
            pipe.multi()
            pipe.set(self._key_best_trial(study_id), pickle.dumps(trial_id))

            study_summary = self._get_study_summary(study_id)
            study_summary.best_trial = self.get_trial(trial_id)
            pipe.set(self._key_study_summary(study_id), pickle.dumps(study_summary))
            pipe.execute()

    def get_trial_param(self, trial_id, param_name):
        # type: (int, str) -> float

        distribution = self.get_trial(trial_id).distributions[param_name]
        return distribution.to_internal_repr(self.get_trial(trial_id).params[param_name])

    def set_trial_value(self, trial_id, value):
        # type: (int, float) -> None

        self._check_trial_id(trial_id)
        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)

        trial.value = value
        self._redis.set(self._key_trial(trial_id), pickle.dumps(trial))

    def _update_cache(self, trial_id):
        # type: (int) -> None

        trial = self.get_trial(trial_id)
        if trial.state != TrialState.COMPLETE:
            return
        study_id = self.get_study_id_from_trial_id(trial_id)
        if not self._redis.exists("study_id:{:010d}:best_trial_id".format(study_id)):
            self._set_best_trial(study_id, trial_id)
            return

        best_value_or_none = self.get_best_trial(study_id).value
        assert best_value_or_none is not None
        assert trial.value is not None
        best_value = float(best_value_or_none)
        new_value = float(trial.value)

        # Complete trials do not have `None` values.
        assert new_value is not None

        if self.get_study_direction(study_id) == StudyDirection.MAXIMIZE:
            if new_value > best_value:
                self._set_best_trial(study_id, trial_id)
        else:
            if new_value < best_value:
                self._set_best_trial(study_id, trial_id)

        return

    def set_trial_intermediate_value(self, trial_id, step, intermediate_value):
        # type: (int, int, float) -> bool

        self._check_trial_id(trial_id)
        self.check_trial_is_updatable(trial_id, self.get_trial(trial_id).state)

        frozen_trial = self.get_trial(trial_id)
        if step in frozen_trial.intermediate_values:
            return False

        frozen_trial.intermediate_values[step] = intermediate_value
        self._set_trial(trial_id, frozen_trial)

        return True

    def set_trial_user_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        self._check_trial_id(trial_id)
        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)
        trial.user_attrs[key] = value
        self._set_trial(trial_id, trial)

    def set_trial_system_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        self._check_trial_id(trial_id)
        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)
        trial.system_attrs[key] = value
        self._set_trial(trial_id, trial)

    @staticmethod
    def _key_trial(trial_id):
        # type: (int) -> str

        return "trial_id:{:010d}:frozentrial".format(trial_id)

    def get_trial(self, trial_id):
        # type: (int) -> FrozenTrial

        self._check_trial_id(trial_id)

        frozen_trial_pkl = self._redis.get(self._key_trial(trial_id))
        assert frozen_trial_pkl is not None
        return pickle.loads(frozen_trial_pkl)

    def _set_trial(self, trial_id, trial):
        # type: (int, FrozenTrial) -> None

        self._redis.set(self._key_trial(trial_id), pickle.dumps(trial))

    def _del_trial(self, trial_id):
        # type: (int) -> None

        with self._redis.pipeline() as pipe:
            pipe.multi()
            pipe.delete(self._key_trial(trial_id))
            pipe.delete("trial_id:{:010d}:study_id".format(trial_id))
            pipe.execute()

    def _get_study_trials(self, study_id):
        # type: (int) -> List[int]

        self._check_study_id(study_id)

        study_trial_list_key = "study_id:{:010d}:trial_list".format(study_id)
        return [int(tid) for tid in self._redis.lrange(study_trial_list_key, 0, -1)]

    def get_all_trials(self, study_id, deepcopy=True):
        # type: (int, bool) -> List[FrozenTrial]

        self._check_study_id(study_id)

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
        # type: (int, Optional[TrialState]) -> int

        self._check_study_id(study_id)
        if state is None:
            return len(self.get_all_trials(study_id))

        return len([t for t in self.get_all_trials(study_id) if t.state == state])

    def _check_study_id(self, study_id):
        # type: (int) -> None

        if not self._redis.exists("study_id:{:010d}:study_name".format(study_id)):
            raise KeyError("study_id {} does not exist.".format(study_id))

    def _check_trial_id(self, trial_id: int) -> None:

        if not self._redis.exists(self._key_trial(trial_id)):
            raise KeyError("study_id {} does not exist.".format(trial_id))
