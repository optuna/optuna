from collections.abc import Mapping
import copy
from datetime import datetime
import pickle
from typing import Any
from typing import Callable
from typing import cast
from typing import Container
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Union

import optuna
from optuna import distributions
from optuna import exceptions
from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna.storages import BaseStorage
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.storages._heartbeat import BaseHeartbeat
from optuna.study._frozen import FrozenStudy
from optuna.study._study_direction import StudyDirection
from optuna.study._study_summary import StudySummary
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = optuna.logging.get_logger(__name__)


with try_import() as _imports:
    import redis


@experimental_class("1.4.0")
class RedisStorage(BaseStorage, BaseHeartbeat):
    """Storage class for Redis backend.

    Note that library users can instantiate this class, but the attributes
    provided by this class are not supposed to be directly accessed by them.

    Example:

        We create an :class:`~optuna.storages.RedisStorage` instance using
        the given redis database URL.

        .. code::

            import optuna


            def objective(trial):
                ...


            storage = optuna.storages.RedisStorage(
                url="redis://passwd@localhost:port/db",
            )

            study = optuna.create_study(storage=storage)
            study.optimize(objective)

    Args:
        url:
            URL of the redis storage, password and db are optional. (ie: redis://localhost:6379)
        heartbeat_interval:
            Interval to record the heartbeat. It is recorded every ``interval`` seconds.
            ``heartbeat_interval`` must be :obj:`None` or a positive integer.

            .. note::
                The heartbeat is supposed to be used with :meth:`~optuna.study.Study.optimize`.
                If you use :meth:`~optuna.study.Study.ask` and
                :meth:`~optuna.study.Study.tell` instead, it will not work.

        grace_period:
            Grace period before a running trial is failed from the last heartbeat.
            ``grace_period`` must be :obj:`None` or a positive integer.
            If it is :obj:`None`, the grace period will be `2 * heartbeat_interval`.
        failed_trial_callback:
            A callback function that is invoked after failing each stale trial.
            The function must accept two parameters with the following types in this order:
            :class:`~optuna.study.Study` and :class:`~optuna.trial.FrozenTrial`.

            .. note::
                The procedure to fail existing stale trials is called just before asking the
                study for a new trial.

    .. note::
        If you use plan to use Redis as a storage mechanism for optuna,
        make sure Redis in installed and running.
        Please execute ``$ pip install -U redis`` to install redis python library.

    """

    def __init__(
        self,
        url: str,
        *,
        heartbeat_interval: Optional[int] = None,
        grace_period: Optional[int] = None,
        failed_trial_callback: Optional[Callable[["optuna.Study", FrozenTrial], None]] = None,
    ) -> None:

        _imports.check()

        if heartbeat_interval is not None and heartbeat_interval <= 0:
            raise ValueError("The value of `heartbeat_interval` should be a positive integer.")
        if grace_period is not None and grace_period <= 0:
            raise ValueError("The value of `grace_period` should be a positive integer.")

        self._url = url
        self._redis = redis.Redis.from_url(url)
        self.heartbeat_interval = heartbeat_interval
        self.grace_period = grace_period
        self.failed_trial_callback = failed_trial_callback

    def create_new_study(self, study_name: Optional[str] = None) -> int:

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
                "study_id:{:010d}:directions".format(study_id),
                pickle.dumps([StudyDirection.NOT_SET]),
            )
            # TODO(wattlebirdaz): Replace StudySummary with FrozenStudy.
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

        _logger.info("A new study created in Redis with name: {}".format(study_name))

        return study_id

    def delete_study(self, study_id: int) -> None:

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
            pipe.delete("study_id:{:010d}:directions".format(study_id))
            pipe.delete("study_id:{:010d}:best_trial_id".format(study_id))
            pipe.delete("study_id:{:010d}:params_distribution".format(study_id))
            pipe.execute()

    @staticmethod
    def _key_study_name(study_name: str) -> str:

        return "study_name:{}:study_id".format(study_name)

    @staticmethod
    def _key_study_summary(study_id: int) -> str:

        return "study_id:{:010d}:study_summary".format(study_id)

    def _set_study_summary(self, study_id: int, study_summary: StudySummary) -> None:

        self._redis.set(self._key_study_summary(study_id), pickle.dumps(study_summary))

    def _get_study_summary(self, study_id: int, include_best_trial: bool = True) -> StudySummary:

        summary_pkl = self._redis.get(self._key_study_summary(study_id))
        assert summary_pkl is not None
        summary = pickle.loads(summary_pkl)
        if not include_best_trial:
            summary.best_trial = None
        return summary

    def _del_study_summary(self, study_id: int) -> None:

        self._redis.delete(self._key_study_summary(study_id))

    @staticmethod
    def _key_study_direction(study_id: int) -> str:

        return "study_id:{:010d}:directions".format(study_id)

    def set_study_directions(self, study_id: int, directions: Sequence[StudyDirection]) -> None:

        self._check_study_id(study_id)

        if self._redis.exists(self._key_study_direction(study_id)):
            direction_pkl = self._redis.get(self._key_study_direction(study_id))
            assert direction_pkl is not None
            current_directions = pickle.loads(direction_pkl)
            if (
                current_directions[0] != StudyDirection.NOT_SET
                and current_directions != directions
            ):
                raise ValueError(
                    "Cannot overwrite study direction from {} to {}.".format(
                        current_directions, directions
                    )
                )

        queries: Mapping[Union[str, bytes], Union[bytes, float, int, str]]
        queries = dict()

        queries[self._key_study_direction(study_id)] = pickle.dumps(directions)
        study_summary = self._get_study_summary(study_id)
        study_summary._directions = list(directions)
        queries[self._key_study_summary(study_id)] = pickle.dumps(study_summary)

        self._redis.mset(queries)

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:

        self._check_study_id(study_id)

        study_summary = self._get_study_summary(study_id)
        study_summary.user_attrs[key] = value
        self._set_study_summary(study_id, study_summary)

    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:

        self._check_study_id(study_id)

        study_summary = self._get_study_summary(study_id)
        study_summary.system_attrs[key] = value
        self._set_study_summary(study_id, study_summary)

    def get_study_id_from_name(self, study_name: str) -> int:

        if not self._redis.exists(self._key_study_name(study_name)):
            raise KeyError("No such study {}.".format(study_name))
        study_id_pkl = self._redis.get(self._key_study_name(study_name))
        assert study_id_pkl is not None
        return pickle.loads(study_id_pkl)

    def _get_study_id_from_trial_id(self, trial_id: int) -> int:

        study_id_pkl = self._redis.get("trial_id:{:010d}:study_id".format(trial_id))
        if study_id_pkl is None:
            raise KeyError("No such trial: {}.".format(trial_id))
        return pickle.loads(study_id_pkl)

    def get_study_name_from_id(self, study_id: int) -> str:

        self._check_study_id(study_id)

        study_name_pkl = self._redis.get("study_id:{:010d}:study_name".format(study_id))
        if study_name_pkl is None:
            raise KeyError("No such study: {}.".format(study_id))
        return pickle.loads(study_name_pkl)

    def get_study_directions(self, study_id: int) -> List[StudyDirection]:

        direction_pkl = self._redis.get("study_id:{:010d}:directions".format(study_id))
        if direction_pkl is None:
            raise KeyError("No such study: {}.".format(study_id))
        return list(pickle.loads(direction_pkl))

    def get_study_user_attrs(self, study_id: int) -> Dict[str, Any]:

        self._check_study_id(study_id)

        study_summary = self._get_study_summary(study_id)
        return study_summary.user_attrs

    def get_study_system_attrs(self, study_id: int) -> Dict[str, Any]:

        self._check_study_id(study_id)

        study_summary = self._get_study_summary(study_id)
        return study_summary.system_attrs

    @staticmethod
    def _key_study_param_distribution(study_id: int) -> str:

        return "study_id:{:010d}:params_distribution".format(study_id)

    def _get_study_param_distribution(self, study_id: int) -> Dict:

        if self._redis.exists(self._key_study_param_distribution(study_id)):
            param_distribution_pkl = self._redis.get(self._key_study_param_distribution(study_id))
            assert param_distribution_pkl is not None
            return pickle.loads(param_distribution_pkl)
        else:
            return {}

    def _set_study_param_distribution(self, study_id: int, param_distribution: Dict) -> None:

        self._redis.set(
            self._key_study_param_distribution(study_id), pickle.dumps(param_distribution)
        )

    def get_all_studies(self) -> List[FrozenStudy]:
        queries = []
        study_ids = [pickle.loads(sid) for sid in self._redis.lrange("study_list", 0, -1)]
        for study_id in study_ids:
            queries.append(self._key_study_summary(study_id))

        frozen_studies = []
        summary_pkls = self._redis.mget(queries)
        for summary_pkl in summary_pkls:
            assert summary_pkl is not None
            summary = pickle.loads(summary_pkl)
            frozen_studies.append(
                FrozenStudy(
                    study_name=summary.study_name,
                    direction=summary.direction,
                    user_attrs=summary.user_attrs,
                    system_attrs=summary.system_attrs,
                    study_id=summary._study_id,
                    directions=summary.directions,
                )
            )

        return frozen_studies

    def create_new_trial(self, study_id: int, template_trial: Optional[FrozenTrial] = None) -> int:

        return self._create_new_trial(study_id, template_trial)._trial_id

    def _create_new_trial(
        self, study_id: int, template_trial: Optional[FrozenTrial] = None
    ) -> FrozenTrial:
        """Create a new trial and returns a :class:`~optuna.trial.FrozenTrial`.

        Args:
            study_id:
                Study id.
            template_trial:
                A :class:`~optuna.trial.FrozenTrial` with default values for trial attributes.

        Returns:
            A :class:`~optuna.trial.FrozenTrial` instance.

        """

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
            min_datetime_start = min(
                [
                    t.datetime_start
                    for t in self.get_all_trials(study_id)
                    if t.datetime_start is not None
                ],
                default=None,
            )
            study_summary.datetime_start = min_datetime_start
            pipe.set(self._key_study_summary(study_id), pickle.dumps(study_summary))
            pipe.execute()

        if trial.state.is_finished():
            self._update_cache(trial_id)

        return trial

    @staticmethod
    def _create_running_trial() -> FrozenTrial:

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

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: distributions.BaseDistribution,
    ) -> None:

        self.check_trial_is_updatable(trial_id, self.get_trial(trial_id).state)

        # Check param distribution compatibility with previous trial(s).
        study_id = self._get_study_id_from_trial_id(trial_id)
        param_distribution = self._get_study_param_distribution(study_id)
        if param_name in param_distribution:
            distributions.check_distribution_compatibility(
                param_distribution[param_name], distribution
            )

        trial = self.get_trial(trial_id)

        queries: Mapping[Union[str, bytes], Union[bytes, float, int, str]]
        queries = dict()

        # Set study param distribution.
        param_distribution[param_name] = distribution
        queries[self._key_study_param_distribution(study_id)] = pickle.dumps(param_distribution)

        # Set params.
        trial.params[param_name] = distribution.to_external_repr(param_value_internal)
        trial.distributions[param_name] = distribution
        queries[self._key_trial(trial_id)] = pickle.dumps(trial)

        self._redis.mset(queries)

    def get_trial_id_from_study_id_trial_number(self, study_id: int, trial_number: int) -> int:

        trial_ids = self._get_study_trials(study_id)
        if len(trial_ids) <= trial_number:
            raise KeyError(
                "No trial with trial number {} exists in study with study_id {}.".format(
                    trial_number, study_id
                )
            )

        return trial_ids[trial_number]

    @staticmethod
    def _key_best_trial(study_id: int) -> str:

        return "study_id:{:010d}:best_trial_id".format(study_id)

    def get_best_trial(self, study_id: int) -> FrozenTrial:

        if not self._redis.exists(self._key_best_trial(study_id)):
            all_trials = self.get_all_trials(study_id, deepcopy=False)
            all_trials = [t for t in all_trials if t.state is TrialState.COMPLETE]

            if len(all_trials) == 0:
                raise ValueError("No trials are completed yet.")

            _direction = self.get_study_directions(study_id)
            if len(_direction) > 1:
                raise RuntimeError(
                    "Best trial can be obtained only for single-objective optimization."
                )
            direction = _direction[0]

            if direction == StudyDirection.MAXIMIZE:
                best_trial = max(all_trials, key=lambda t: cast(float, t.value))
            else:
                best_trial = min(all_trials, key=lambda t: cast(float, t.value))

            self._set_best_trial(study_id, best_trial.number)
        else:
            best_trial_id_pkl = self._redis.get(self._key_best_trial(study_id))
            assert best_trial_id_pkl is not None
            best_trial_id = pickle.loads(best_trial_id_pkl)
            best_trial = self.get_trial(best_trial_id)

        return best_trial

    def _set_best_trial(self, study_id: int, trial_id: int) -> None:

        queries: Mapping[Union[str, bytes], Union[bytes, float, int, str]]
        queries = dict()

        queries[self._key_best_trial(study_id)] = pickle.dumps(trial_id)
        study_summary = self._get_study_summary(study_id)
        study_summary.best_trial = self.get_trial(trial_id)
        queries[self._key_study_summary(study_id)] = pickle.dumps(study_summary)

        self._redis.mset(queries)

    def _check_and_set_param_distribution(
        self,
        study_id: int,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: distributions.BaseDistribution,
    ) -> None:

        self._check_study_id(study_id)
        self.set_trial_param(trial_id, param_name, param_value_internal, distribution)

    def set_trial_state_values(
        self, trial_id: int, state: TrialState, values: Optional[Sequence[float]] = None
    ) -> bool:

        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)

        if state == TrialState.RUNNING and trial.state != TrialState.WAITING:
            return False

        trial.state = state
        if values is not None:
            trial.values = values

        if state == TrialState.RUNNING:
            trial.datetime_start = datetime.now()

        if state.is_finished():
            trial.datetime_complete = datetime.now()
            self._redis.set(self._key_trial(trial_id), pickle.dumps(trial))
            self._update_cache(trial_id)

            # To ensure that there are no failed trials with heartbeats in the DB
            # under any circumstances
            study_id = self._get_study_id_from_trial_id(trial_id)
            self._redis.hdel(self._key_study_heartbeats(study_id), str(trial_id))
        else:
            self._redis.set(self._key_trial(trial_id), pickle.dumps(trial))

        return True

    def _update_cache(self, trial_id: int) -> None:

        trial = self.get_trial(trial_id)
        if trial.state != TrialState.COMPLETE:
            return
        study_id = self._get_study_id_from_trial_id(trial_id)

        _direction = self.get_study_directions(study_id)
        if len(_direction) > 1:
            return
        direction = _direction[0]

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

        if direction == StudyDirection.MAXIMIZE:
            if new_value > best_value:
                self._set_best_trial(study_id, trial_id)
        else:
            if new_value < best_value:
                self._set_best_trial(study_id, trial_id)

        return

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:

        frozen_trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, frozen_trial.state)
        frozen_trial.intermediate_values[step] = intermediate_value
        self._set_trial(trial_id, frozen_trial)

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:

        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)
        trial.user_attrs[key] = value
        self._set_trial(trial_id, trial)

    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:

        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)
        trial.system_attrs[key] = value
        self._set_trial(trial_id, trial)

    @staticmethod
    def _key_trial(trial_id: int) -> str:

        return "trial_id:{:010d}:frozentrial".format(trial_id)

    def get_trial(self, trial_id: int) -> FrozenTrial:

        self._check_trial_id(trial_id)

        frozen_trial_pkl = self._redis.get(self._key_trial(trial_id))
        assert frozen_trial_pkl is not None
        return pickle.loads(frozen_trial_pkl)

    def _set_trial(self, trial_id: int, trial: FrozenTrial) -> None:

        self._redis.set(self._key_trial(trial_id), pickle.dumps(trial))

    def _get_study_trials(self, study_id: int) -> List[int]:

        self._check_study_id(study_id)

        study_trial_list_key = "study_id:{:010d}:trial_list".format(study_id)
        return [int(tid) for tid in self._redis.lrange(study_trial_list_key, 0, -1)]

    def get_all_trials(
        self,
        study_id: int,
        deepcopy: bool = True,
        states: Optional[Container[TrialState]] = None,
    ) -> List[FrozenTrial]:
        # Redis will copy data from storage with `mget`.
        # No further copying is required even if the deepcopy option is true.
        return self._get_trials(study_id, states, set())

    def _get_trials(
        self,
        study_id: int,
        states: Optional[Container[TrialState]],
        excluded_trial_ids: Set[int],
    ) -> List[FrozenTrial]:

        self._check_study_id(study_id)

        queries = []
        trial_ids = set(self._get_study_trials(study_id)) - excluded_trial_ids
        for trial_id in trial_ids:
            queries.append(self._key_trial(trial_id))

        trials = []
        frozen_trial_pkls = self._redis.mget(queries)
        for frozen_trial_pkl in frozen_trial_pkls:
            assert frozen_trial_pkl is not None
            frozen_trial = pickle.loads(frozen_trial_pkl)

            if states is None or frozen_trial.state in states:
                trials.append(frozen_trial)

        return trials

    def _check_study_id(self, study_id: int) -> None:

        if not self._redis.exists("study_id:{:010d}:study_name".format(study_id)):
            raise KeyError("study_id {} does not exist.".format(study_id))

    def _check_trial_id(self, trial_id: int) -> None:

        if not self._redis.exists(self._key_trial(trial_id)):
            raise KeyError("trial_id {} does not exist.".format(trial_id))

    def record_heartbeat(self, trial_id: int) -> None:
        study_id = self._get_study_id_from_trial_id(trial_id)
        self._redis.hset(
            self._key_study_heartbeats(study_id),
            str(trial_id),
            pickle.dumps(self._get_redis_time()),
        )

    def _get_stale_trial_ids(self, study_id: int) -> List[int]:
        assert self.heartbeat_interval is not None

        if self.grace_period is None:
            grace_period = 2 * self.heartbeat_interval
        else:
            grace_period = self.grace_period

        current_time = self._get_redis_time()
        heartbeats = self._redis.hgetall(self._key_study_heartbeats(study_id))
        stale = []
        for trial_id_raw, last_heartbeat_raw in heartbeats.items():
            last_heartbeat = pickle.loads(last_heartbeat_raw)
            if current_time - last_heartbeat > grace_period:
                trial_id = int(trial_id_raw)
                stale.append(trial_id)
        return stale

    def _get_redis_time(self) -> float:
        seconds, microseconds = self._redis.time()
        return seconds + microseconds * 1e-6

    def get_heartbeat_interval(self) -> Optional[int]:
        return self.heartbeat_interval

    @staticmethod
    def _key_study_heartbeats(study_id: int) -> str:
        return "study_id:{:010d}:heartbeats".format(study_id)

    def get_failed_trial_callback(self) -> Optional[Callable[["optuna.Study", FrozenTrial], None]]:
        return self.failed_trial_callback
