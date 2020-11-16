import asyncio
import datetime
import inspect
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
import uuid

import optuna
from optuna._experimental import experimental
from optuna._imports import try_import
from optuna.distributions import BaseDistribution
from optuna.distributions import distribution_to_json
from optuna.distributions import json_to_distribution
from optuna.storages import BaseStorage
from optuna.study import StudyDirection
from optuna.study import StudySummary
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


with try_import() as _imports:
    import distributed
    from distributed.utils import thread_state
    from distributed.worker import get_client

# Serialization utilities


def _serialize_datetime(dt: datetime.datetime) -> dict:
    return {"__datetime__": True, "as_str": dt.strftime("%Y%m%dT%H:%M:%S.%f")}


def _deserialize_datetime(data: dict) -> datetime.datetime:
    return datetime.datetime.strptime(data["as_str"], "%Y%m%dT%H:%M:%S.%f")


def _serialize_frozentrial(trial: FrozenTrial) -> dict:
    data = trial.__dict__.copy()
    data["state"] = data["state"].name
    for attr in [
        "trial_id",
        "number",
        "params",
        "user_attrs",
        "system_attrs",
        "distributions",
        "datetime_start",
    ]:
        data[attr] = data.pop(f"_{attr}")
    data["distributions"] = {k: distribution_to_json(v) for k, v in data["distributions"].items()}
    if data["datetime_start"] is not None:
        data["datetime_start"] = _serialize_datetime(data["datetime_start"])
    if data["datetime_complete"] is not None:
        data["datetime_complete"] = _serialize_datetime(data["datetime_complete"])
    return data


def _deserialize_frozentrial(data: dict) -> FrozenTrial:
    data["state"] = getattr(TrialState, data["state"])
    data["distributions"] = {k: json_to_distribution(v) for k, v in data["distributions"].items()}
    if data["datetime_start"] is not None:
        data["datetime_start"] = _deserialize_datetime(data["datetime_start"])
    if data["datetime_complete"] is not None:
        data["datetime_complete"] = _deserialize_datetime(data["datetime_complete"])
    trail = FrozenTrial(**data)
    return trail


def _serialize_studysummary(summary: StudySummary) -> dict:
    data = summary.__dict__.copy()
    data["study_id"] = data.pop("_study_id")
    data["best_trial"] = _serialize_frozentrial(data["best_trial"])
    data["datetime_start"] = _serialize_datetime(data["datetime_start"])
    data["direction"] = data["direction"]["name"]
    return data


def _deserialize_studysummary(data: dict) -> StudySummary:
    data["direction"] = getattr(StudyDirection, data["direction"])
    data["best_trial"] = _deserialize_frozentrial(data["best_trial"])
    data["datetime_start"] = _deserialize_datetime(data["datetime_start"])
    summary = StudySummary(**data)
    return summary


def _serialize_studydirection(direction: StudyDirection) -> str:
    return direction.name


def _deserialize_studydirection(data: str) -> StudyDirection:
    return getattr(StudyDirection, data)


class _OptunaSchedulerExtension:
    def __init__(self, scheduler: "distributed.Scheduler"):
        self.scheduler = scheduler
        self.storages: Dict[str, BaseStorage] = {}

        self.scheduler.handlers.update(
            {
                "optuna_create_new_study": self.create_new_study,
                "optuna_delete_study": self.delete_study,
                "optuna_set_study_user_attr": self.set_study_user_attr,
                "optuna_set_study_system_attr": self.set_study_system_attr,
                "optuna_set_study_direction": self.set_study_direction,
                "optuna_get_study_id_from_name": self.get_study_id_from_name,
                "optuna_get_study_id_from_trial_id": self.get_study_id_from_trial_id,
                "optuna_get_study_name_from_id": self.get_study_name_from_id,
                "optuna_read_trials_from_remote_storage": self.read_trials_from_remote_storage,
                "optuna_get_study_direction": self.get_study_direction,
                "optuna_get_study_user_attrs": self.get_study_user_attrs,
                "optuna_get_study_system_attrs": self.get_study_system_attrs,
                "optuna_get_all_study_summaries": self.get_all_study_summaries,
                "optuna_create_new_trial": self.create_new_trial,
                "optuna_set_trial_state": self.set_trial_state,
                "optuna_set_trial_param": self.set_trial_param,
                "optuna_get_trial_number_from_id": self.get_trial_number_from_id,
                "optuna_get_trial_param": self.get_trial_param,
                "optuna_set_trial_value": self.set_trial_value,
                "optuna_set_trial_intermediate_value": self.set_trial_intermediate_value,
                "optuna_set_trial_user_attr": self.set_trial_user_attr,
                "optuna_set_trial_system_attr": self.set_trial_system_attr,
                "optuna_get_trial": self.get_trial,
                "optuna_get_all_trials": self.get_all_trials,
                "optuna_get_n_trials": self.get_n_trials,
            }
        )

        self.scheduler.extensions["optuna"] = self

    def get_storage(self, name: str) -> BaseStorage:
        return self.storages[name]

    def create_new_study(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_name: Optional[str] = None,
    ) -> int:
        return self.get_storage(storage_name).create_new_study(study_name=study_name)

    def delete_study(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
    ) -> None:
        return self.get_storage(storage_name).delete_study(study_id=study_id)

    def set_study_user_attr(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
        key: str,
        value: Any,
    ) -> None:
        return self.get_storage(storage_name).set_study_user_attr(
            study_id=study_id, key=key, value=value
        )

    def set_study_system_attr(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
        key: str,
        value: Any,
    ) -> None:
        return self.get_storage(storage_name).set_study_system_attr(
            study_id=study_id,
            key=key,
            value=value,
        )

    def set_study_direction(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
        direction: str,
    ) -> None:
        return self.get_storage(storage_name).set_study_direction(
            study_id=study_id,
            direction=_deserialize_studydirection(direction),
        )

    def get_study_id_from_name(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_name: str,
    ) -> int:
        return self.get_storage(storage_name).get_study_id_from_name(study_name=study_name)

    def get_study_id_from_trial_id(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        trial_id: int,
    ) -> int:
        return self.get_storage(storage_name).get_study_id_from_trial_id(trial_id=trial_id)

    def get_study_name_from_id(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
    ) -> str:
        return self.get_storage(storage_name).get_study_name_from_id(study_id=study_id)

    def get_study_direction(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
    ) -> str:
        direction = self.get_storage(storage_name).get_study_direction(study_id=study_id)
        return _serialize_studydirection(direction)

    def get_study_user_attrs(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
    ) -> Dict[str, Any]:
        return self.get_storage(storage_name).get_study_user_attrs(study_id=study_id)

    def get_study_system_attrs(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
    ) -> Dict[str, Any]:
        return self.get_storage(storage_name).get_study_system_attrs(study_id=study_id)

    def get_all_study_summaries(
        self, comm: "distributed.comm.tcp.TCP", storage_name: str
    ) -> List[dict]:
        summaries = self.get_storage(storage_name).get_all_study_summaries()
        return [_serialize_studysummary(s) for s in summaries]

    def create_new_trial(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
        template_trial: Optional[FrozenTrial] = None,
    ) -> int:
        return self.get_storage(storage_name).create_new_trial(
            study_id=study_id,
            template_trial=template_trial,
        )

    def set_trial_state(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        trial_id: int,
        state: str,
    ) -> bool:
        return self.get_storage(storage_name).set_trial_state(
            trial_id=trial_id,
            state=getattr(TrialState, state),
        )

    def set_trial_param(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: str,
    ) -> None:
        return self.get_storage(storage_name).set_trial_param(
            trial_id=trial_id,
            param_name=param_name,
            param_value_internal=param_value_internal,
            distribution=json_to_distribution(distribution),
        )

    def get_trial_number_from_id(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        trial_id: int,
    ) -> int:
        return self.get_storage(storage_name).get_trial_number_from_id(trial_id=trial_id)

    def get_trial_param(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        trial_id: int,
        param_name: str,
    ) -> float:
        return self.get_storage(storage_name).get_trial_param(
            trial_id=trial_id,
            param_name=param_name,
        )

    def set_trial_value(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        trial_id: int,
        value: float,
    ) -> None:
        return self.get_storage(storage_name).set_trial_value(
            trial_id=trial_id,
            value=value,
        )

    def set_trial_intermediate_value(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        trial_id: int,
        step: int,
        intermediate_value: float,
    ) -> None:
        return self.get_storage(storage_name).set_trial_intermediate_value(
            trial_id=trial_id,
            step=step,
            intermediate_value=intermediate_value,
        )

    def set_trial_user_attr(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        trial_id: int,
        key: str,
        value: Any,
    ) -> None:
        return self.get_storage(storage_name).set_trial_user_attr(
            trial_id=trial_id,
            key=key,
            value=value,
        )

    def set_trial_system_attr(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        trial_id: int,
        key: str,
        value: Any,
    ) -> None:
        return self.get_storage(storage_name).set_trial_system_attr(
            trial_id=trial_id,
            key=key,
            value=value,
        )

    def get_trial(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        trial_id: int,
    ) -> dict:
        trial = self.get_storage(storage_name).get_trial(trial_id=trial_id)
        return _serialize_frozentrial(trial)

    def get_all_trials(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
        deepcopy: bool = True,
    ) -> List[dict]:
        trials = self.get_storage(storage_name).get_all_trials(
            study_id=study_id,
            deepcopy=deepcopy,
        )
        return [_serialize_frozentrial(t) for t in trials]

    def get_n_trials(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
        state: Optional[TrialState] = None,
    ) -> int:
        return self.get_storage(storage_name).get_n_trials(
            study_id=study_id,
            state=state,
        )

    def read_trials_from_remote_storage(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
    ) -> None:
        return self.get_storage(storage_name).read_trials_from_remote_storage(study_id=study_id)


def _register_with_scheduler(
    dask_scheduler: "distributed.Scheduler", storage: str, name: str
) -> None:
    if "optuna" not in dask_scheduler.extensions:
        ext = _OptunaSchedulerExtension(dask_scheduler)
    else:
        ext = dask_scheduler.extensions["optuna"]

    if name not in ext.storages:
        ext.storages[name] = optuna.storages.get_storage(storage)


def _use_basestorage_doc(func: Callable) -> Callable:
    method = getattr(optuna.storages.BaseStorage, func.__name__, None)
    if method is not None:
        # Ensure BaseStorage and DaskStorage have the same signature
        assert inspect.signature(func) == inspect.signature(method)
        # Overwrite docstring if one does not exist already
        if func.__doc__ is not None:
            func.__doc__ = method.__doc__
    return func


@experimental("2.4.0")
class DaskStorage(optuna.storages.BaseStorage):
    """Dask-compatible storage class.

    This storage class wraps a Optuna storage class (e.g. Optuna’s in-memory or sqlite storage)
    and is used to run optimization trials in parallel on a Dask cluster.
    The underlying Optuna storage object lives on the cluster’s scheduler and any method calls on
    the ``DaskStorage`` instance results in the same method being called on the underlying Optuna
    storage object.

    See `this example <https://github.com/optuna/optuna/blob/master/
    examples/dask_simple.py>`__
    for how to use ``DaskStorage`` to extend Optuna's in-memory storage class to run across
    multiple processes.

    Args:
        storage:
            Optuna storage url to use for underlying Optuna storage class to wrap
            (e.g. ``None`` for in-memory storage, ``sqlite:///example.db`` for SQLite storage).
            Defaults to ``None``.

        name:
            Unique identifier for the Dask storage class. If not provided, a random name
            will be generated.

        client:
            Dask ``Client`` to connect to. If not provided, will attempt to find an
            existing ``Client``.

    """

    def __init__(self, storage: str = None, name: str = None, client: "distributed.Client" = None):
        _imports.check()
        self.name = name or f"dask-storage-{uuid.uuid4().hex}"
        self.client = client or get_client()

        if self.client.asynchronous or getattr(thread_state, "on_event_loop_thread", False):

            async def _register() -> DaskStorage:
                await self.client.run_on_scheduler(
                    _register_with_scheduler, storage=storage, name=self.name
                )
                return self

            self._started = asyncio.ensure_future(_register())
        else:
            self.client.run_on_scheduler(_register_with_scheduler, storage=storage, name=self.name)

    def __await__(self) -> Generator[Any, None, "DaskStorage"]:
        if hasattr(self, "_started"):
            return self._started.__await__()
        else:

            async def _() -> DaskStorage:
                return self

            return _().__await__()

    def __reduce__(self) -> tuple:
        return (DaskStorage, (None, self.name))

    def get_base_storage(self) -> BaseStorage:
        def _(dask_scheduler: distributed.Scheduler, name: str = None) -> BaseStorage:
            return dask_scheduler.extensions["optuna"].storages[name]

        return self.client.run_on_scheduler(_, name=self.name)

    @_use_basestorage_doc
    def create_new_study(self, study_name: Optional[str] = None) -> int:
        return self.client.sync(
            self.client.scheduler.optuna_create_new_study,
            storage_name=self.name,
            study_name=study_name,
        )

    @_use_basestorage_doc
    def delete_study(self, study_id: int) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_delete_study,
            storage_name=self.name,
            study_id=study_id,
        )

    @_use_basestorage_doc
    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_study_user_attr,
            storage_name=self.name,
            study_id=study_id,
            key=key,
            value=value,
        )

    @_use_basestorage_doc
    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_study_system_attr,
            storage_name=self.name,
            study_id=study_id,
            key=key,
            value=value,
        )

    @_use_basestorage_doc
    def set_study_direction(self, study_id: int, direction: StudyDirection) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_study_direction,
            storage_name=self.name,
            study_id=study_id,
            direction=direction.name,
        )

    # Basic study access

    @_use_basestorage_doc
    def get_study_id_from_name(self, study_name: str) -> int:
        return self.client.sync(
            self.client.scheduler.optuna_get_study_id_from_name,
            study_name=study_name,
            storage_name=self.name,
        )

    @_use_basestorage_doc
    def get_study_id_from_trial_id(self, trial_id: int) -> int:
        return self.client.sync(
            self.client.scheduler.optuna_get_study_id_from_trial_id,
            storage_name=self.name,
            trial_id=trial_id,
        )

    @_use_basestorage_doc
    def get_study_name_from_id(self, study_id: int) -> str:
        return self.client.sync(
            self.client.scheduler.optuna_get_study_name_from_id,
            storage_name=self.name,
            study_id=study_id,
        )

    @_use_basestorage_doc
    def get_study_direction(self, study_id: int) -> StudyDirection:
        direction = self.client.sync(
            self.client.scheduler.optuna_get_study_direction,
            storage_name=self.name,
            study_id=study_id,
        )
        return _deserialize_studydirection(direction)

    @_use_basestorage_doc
    def get_study_user_attrs(self, study_id: int) -> Dict[str, Any]:
        return self.client.sync(
            self.client.scheduler.optuna_get_study_user_attrs,
            storage_name=self.name,
            study_id=study_id,
        )

    @_use_basestorage_doc
    def get_study_system_attrs(self, study_id: int) -> Dict[str, Any]:
        return self.client.sync(
            self.client.scheduler.optuna_get_study_system_attrs,
            storage_name=self.name,
            study_id=study_id,
        )

    async def _get_all_study_summaries(self) -> List[StudySummary]:
        serialized_summaries = await self.client.scheduler.optuna_get_all_study_summaries(
            storage_name=self.name
        )
        return [_deserialize_studysummary(s) for s in serialized_summaries]

    @_use_basestorage_doc
    def get_all_study_summaries(self) -> List[StudySummary]:
        return self.client.sync(self._get_all_study_summaries)

    # Basic trial manipulation

    @_use_basestorage_doc
    def create_new_trial(self, study_id: int, template_trial: Optional[FrozenTrial] = None) -> int:
        return self.client.sync(
            self.client.scheduler.optuna_create_new_trial,
            storage_name=self.name,
            study_id=study_id,
            template_trial=template_trial,
        )

    @_use_basestorage_doc
    def set_trial_state(self, trial_id: int, state: TrialState) -> bool:
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_state,
            storage_name=self.name,
            trial_id=trial_id,
            state=state.name,
        )

    @_use_basestorage_doc
    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
    ) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_param,
            storage_name=self.name,
            trial_id=trial_id,
            param_name=param_name,
            param_value_internal=param_value_internal,
            distribution=distribution_to_json(distribution),
        )

    @_use_basestorage_doc
    def get_trial_number_from_id(self, trial_id: int) -> int:
        return self.client.sync(
            self.client.scheduler.optuna_get_trial_number_from_id,
            storage_name=self.name,
            trial_id=trial_id,
        )

    @_use_basestorage_doc
    def get_trial_param(self, trial_id: int, param_name: str) -> float:
        return self.client.sync(
            self.client.scheduler.optuna_get_trial_param,
            storage_name=self.name,
            trial_id=trial_id,
            param_name=param_name,
        )

    @_use_basestorage_doc
    def set_trial_value(self, trial_id: int, value: float) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_value,
            trial_id=trial_id,
            value=value,
            storage_name=self.name,
        )

    @_use_basestorage_doc
    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_intermediate_value,
            storage_name=self.name,
            trial_id=trial_id,
            step=step,
            intermediate_value=intermediate_value,
        )

    @_use_basestorage_doc
    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_user_attr,
            storage_name=self.name,
            trial_id=trial_id,
            key=key,
            value=value,
        )

    @_use_basestorage_doc
    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_system_attr,
            storage_name=self.name,
            trial_id=trial_id,
            key=key,
            value=value,
        )

    # Basic trial access

    async def _get_trial(self, trial_id: int) -> FrozenTrial:
        serialized_trial = await self.client.scheduler.optuna_get_trial(
            trial_id=trial_id, storage_name=self.name
        )
        return _deserialize_frozentrial(serialized_trial)

    @_use_basestorage_doc
    def get_trial(self, trial_id: int) -> FrozenTrial:
        return self.client.sync(self._get_trial, trial_id=trial_id)

    async def _get_all_trials(self, study_id: int, deepcopy: bool = True) -> List[FrozenTrial]:
        serialized_trials = await self.client.scheduler.optuna_get_all_trials(
            storage_name=self.name,
            study_id=study_id,
            deepcopy=deepcopy,
        )
        return [_deserialize_frozentrial(t) for t in serialized_trials]

    @_use_basestorage_doc
    def get_all_trials(self, study_id: int, deepcopy: bool = True) -> List[FrozenTrial]:
        return self.client.sync(
            self._get_all_trials,
            study_id=study_id,
            deepcopy=deepcopy,
        )

    @_use_basestorage_doc
    def get_n_trials(self, study_id: int, state: Optional[TrialState] = None) -> int:
        return self.client.sync(
            self.client.scheduler.optuna_get_n_trials,
            storage_name=self.name,
            study_id=study_id,
            state=state,
        )

    @_use_basestorage_doc
    def read_trials_from_remote_storage(self, study_id: int) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_read_trials_from_remote_storage,
            storage_name=self.name,
            study_id=study_id,
        )
