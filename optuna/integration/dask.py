import asyncio
import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
import uuid

import optuna
from optuna._experimental import experimental
from optuna._imports import try_import
from optuna.distributions import BaseDistribution
from optuna.distributions import distribution_to_json
from optuna.distributions import json_to_distribution
from optuna.study import StudyDirection
from optuna.study import StudySummary
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


with try_import() as _imports:
    from distributed.utils import thread_state
    from distributed.worker import get_client

# Serialization utilities


def serialize_datetime(obj):
    if isinstance(obj, datetime.datetime):
        return {"__datetime__": True, "as_str": obj.strftime("%Y%m%dT%H:%M:%S.%f")}
    return obj


def deserialize_datetime(obj):
    if "__datetime__" in obj:
        obj = datetime.datetime.strptime(obj["as_str"], "%Y%m%dT%H:%M:%S.%f")
    return obj


def serialize_frozentrial(trial):
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
    data["datetime_start"] = serialize_datetime(data["datetime_start"])
    data["datetime_complete"] = serialize_datetime(data["datetime_complete"])
    return data


def deserialize_frozentrial(data):
    data["state"] = getattr(TrialState, data["state"])
    data["distributions"] = {k: json_to_distribution(v) for k, v in data["distributions"].items()}
    if data["datetime_start"] is not None:
        data["datetime_start"] = deserialize_datetime(data["datetime_start"])
    if data["datetime_complete"] is not None:
        data["datetime_complete"] = deserialize_datetime(data["datetime_complete"])
    trail = FrozenTrial(**data)
    return trail


def serialize_studysummary(summary):
    data = summary.__dict__.copy()
    data["study_id"] = data.pop("_study_id")
    data["best_trial"] = serialize_frozentrial(data["best_trial"])
    data["datetime_start"] = serialize_datetime(data["datetime_start"])
    data["direction"] = data["direction"]["name"]
    return data


def deserialize_studysummary(data):
    data["direction"] = getattr(StudyDirection, data["direction"])
    data["best_trial"] = deserialize_frozentrial(data["best_trial"])
    data["datetime_start"] = deserialize_datetime(data["datetime_start"])
    summary = StudySummary(**data)
    return summary


def serialize_studydirection(direction):
    return direction.name


def deserialize_studydirection(data):
    return getattr(StudyDirection, data)


@experimental("2.4.0")
class OptunaSchedulerExtension:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.storages = {}

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

    def get_storage(self, name):
        return self.storages[name]

    def create_new_study(
        self, comm, study_name: Optional[str] = None, storage_name: str = None
    ) -> int:
        return self.get_storage(storage_name).create_new_study(study_name=study_name)

    def delete_study(self, comm, study_id: int = None, storage_name: str = None) -> None:
        return self.get_storage(storage_name).delete_study(study_id=study_id)

    def set_study_user_attr(
        self, comm, study_id: int, key: str, value: Any, storage_name: str = None
    ) -> None:
        return self.get_storage(storage_name).set_study_user_attr(
            study_id=study_id, key=key, value=value
        )

    def set_study_system_attr(
        self, comm, study_id: int, key: str, value: Any, storage_name: str = None
    ) -> None:
        return self.get_storage(storage_name).set_study_system_attr(
            study_id=study_id,
            key=key,
            value=value,
        )

    def set_study_direction(
        self,
        comm,
        study_id: int,
        direction: StudySummary,
        storage_name: str = None,
    ) -> None:
        return self.get_storage(storage_name).set_study_direction(
            study_id=study_id,
            direction=deserialize_studydirection(direction),
        )

    def get_study_id_from_name(self, comm, study_name: str, storage_name: str = None) -> int:
        return self.get_storage(storage_name).get_study_id_from_name(study_name=study_name)

    def get_study_id_from_trial_id(self, comm, trial_id: int, storage_name: str = None) -> int:
        return self.get_storage(storage_name).get_study_id_from_trial_id(trial_id=trial_id)

    def get_study_name_from_id(self, comm, study_id: int, storage_name: str = None) -> str:
        return self.get_storage(storage_name).get_study_name_from_id(study_id=study_id)

    def get_study_direction(self, comm, study_id: int, storage_name: str = None) -> StudySummary:
        direction = self.get_storage(storage_name).get_study_direction(study_id=study_id)
        return serialize_studydirection(direction)

    def get_study_user_attrs(
        self, comm, study_id: int, storage_name: str = None
    ) -> Dict[str, Any]:
        return self.get_storage(storage_name).get_study_user_attrs(study_id=study_id)

    def get_study_system_attrs(
        self, comm, study_id: int, storage_name: str = None
    ) -> Dict[str, Any]:
        return self.get_storage(storage_name).get_study_system_attrs(study_id=study_id)

    def get_all_study_summaries(self, comm, storage_name: str = None) -> List[StudySummary]:
        summaries = self.get_storage(storage_name).get_all_study_summaries()
        return [serialize_studysummary(s) for s in summaries]

    def create_new_trial(
        self,
        comm,
        study_id: int,
        template_trial: Optional[FrozenTrial] = None,
        storage_name: str = None,
    ) -> int:
        return self.get_storage(storage_name).create_new_trial(
            study_id=study_id,
            template_trial=template_trial,
        )

    def set_trial_state(
        self, comm, trial_id: int, state: TrialState, storage_name: str = None
    ) -> bool:
        return self.get_storage(storage_name).set_trial_state(
            trial_id=trial_id,
            state=getattr(TrialState, state),
        )

    def set_trial_param(
        self,
        comm,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
        storage_name: str = None,
    ) -> None:
        distribution = json_to_distribution(distribution)
        return self.get_storage(storage_name).set_trial_param(
            trial_id=trial_id,
            param_name=param_name,
            param_value_internal=param_value_internal,
            distribution=distribution,
        )

    def get_trial_number_from_id(self, comm, trial_id: int, storage_name: str = None) -> int:
        return self.get_storage(storage_name).get_trial_number_from_id(trial_id=trial_id)

    def get_trial_param(
        self, comm, trial_id: int, param_name: str, storage_name: str = None
    ) -> float:
        return self.get_storage(storage_name).get_trial_param(
            trial_id=trial_id,
            param_name=param_name,
        )

    def set_trial_value(self, comm, trial_id: int, value: float, storage_name: str = None) -> None:
        return self.get_storage(storage_name).set_trial_value(
            trial_id=trial_id,
            value=value,
        )

    def set_trial_intermediate_value(
        self,
        comm,
        trial_id: int,
        step: int,
        intermediate_value: float,
        storage_name: str = None,
    ) -> None:
        return self.get_storage(storage_name).set_trial_intermediate_value(
            trial_id=trial_id,
            step=step,
            intermediate_value=intermediate_value,
        )

    def set_trial_user_attr(
        self, comm, trial_id: int, key: str, value: Any, storage_name: str = None
    ) -> None:
        return self.get_storage(storage_name).set_trial_user_attr(
            trial_id=trial_id,
            key=key,
            value=value,
        )

    def set_trial_system_attr(
        self, comm, trial_id: int, key: str, value: Any, storage_name: str = None
    ) -> None:
        return self.get_storage(storage_name).set_trial_system_attr(
            trial_id=trial_id,
            key=key,
            value=value,
        )

    def get_trial(self, comm, trial_id: int, storage_name: str = None) -> FrozenTrial:
        trial = self.get_storage(storage_name).get_trial(trial_id=trial_id)
        return serialize_frozentrial(trial)

    def get_all_trials(
        self, comm, study_id: int, deepcopy: bool = True, storage_name: str = None
    ) -> List[FrozenTrial]:
        trials = self.get_storage(storage_name).get_all_trials(
            study_id=study_id,
            deepcopy=deepcopy,
        )
        return [serialize_frozentrial(t) for t in trials]

    def get_n_trials(
        self,
        comm,
        study_id: int,
        state: Optional[TrialState] = None,
        storage_name: str = None,
    ) -> int:
        return self.get_storage(storage_name).get_n_trials(
            study_id=study_id,
            state=state,
        )

    def read_trials_from_remote_storage(
        self, comm, study_id: int, storage_name: str = None
    ) -> None:
        return self.get_storage(storage_name).read_trials_from_remote_storage(study_id=study_id)


def register_with_scheduler(dask_scheduler=None, storage=None, name=None):
    if "optuna" not in dask_scheduler.extensions:
        ext = OptunaSchedulerExtension(dask_scheduler)
    else:
        ext = dask_scheduler.extensions["optuna"]

    if name not in ext.storages:
        ext.storages[name] = optuna.storages.get_storage(storage)


def use_basestorage_doc(func):
    method = getattr(optuna.storages.BaseStorage, func.__name__, None)
    if method is not None:
        func.__doc__ = method.__doc__
    return func


@experimental("2.4.0")
class DaskStorage(optuna.storages.BaseStorage):
    """Dask-compatible Storage class

    Parameters
    ----------
    storage
        Optuna storage class or url to use for underlying Optuna storage to wrap
        (e.g. ``None`` for in-memory storage, ``sqlite:///example.db`` for SQLite storage).
        Defaults to ``None``.
    name
        Unique identifier for the Dask storage class. If not provided, a random name
        will be generated.
    client
        Dask ``Client`` to connect to. If not provided, will attempt to find an
        existing ``Client``.
    """

    def __init__(self, storage=None, name: str = None, client=None):
        _imports.check()
        self.name = name or f"dask-storage-{uuid.uuid4().hex}"
        self.client = client or get_client()

        if self.client.asynchronous or getattr(thread_state, "on_event_loop_thread", False):

            async def _register():
                await self.client.run_on_scheduler(
                    register_with_scheduler, storage=storage, name=self.name
                )
                return self

            self._started = asyncio.ensure_future(_register())
        else:
            self.client.run_on_scheduler(register_with_scheduler, storage=storage, name=self.name)

    def __await__(self):
        if hasattr(self, "_started"):
            return self._started.__await__()
        else:

            async def _():
                return self

            return _().__await__()

    def __reduce__(self):
        return (DaskStorage, (None, self.name))

    def get_base_storage(self):
        def _(dask_scheduler=None, name=None):
            return dask_scheduler.extensions["optuna"].storages[name]

        return self.client.run_on_scheduler(_, name=self.name)

    @use_basestorage_doc
    def create_new_study(self, study_name: Optional[str] = None) -> int:
        return self.client.sync(
            self.client.scheduler.optuna_create_new_study,
            study_name=study_name,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def delete_study(self, study_id: int) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_delete_study,
            study_id=study_id,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_study_user_attr,
            study_id=study_id,
            key=key,
            value=value,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_study_system_attr,
            study_id=study_id,
            key=key,
            value=value,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def set_study_direction(self, study_id: int, direction: StudySummary) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_study_direction,
            study_id=study_id,
            direction=direction.name,
            storage_name=self.name,
        )

    # Basic study access

    @use_basestorage_doc
    def get_study_id_from_name(self, study_name: str) -> int:
        return self.client.sync(
            self.client.scheduler.optuna_get_study_id_from_name,
            study_name=study_name,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def get_study_id_from_trial_id(self, trial_id: int) -> int:
        return self.client.sync(
            self.client.scheduler.optuna_get_study_id_from_trial_id,
            trial_id=trial_id,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def get_study_name_from_id(self, study_id: int) -> str:
        return self.client.sync(
            self.client.scheduler.optuna_get_study_name_from_id,
            study_id=study_id,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def get_study_direction(self, study_id: int) -> StudySummary:
        direction = self.client.sync(
            self.client.scheduler.optuna_get_study_direction,
            study_id=study_id,
            storage_name=self.name,
        )
        return deserialize_studydirection(direction)

    @use_basestorage_doc
    def get_study_user_attrs(self, study_id: int) -> Dict[str, Any]:
        return self.client.sync(
            self.client.scheduler.optuna_get_study_user_attrs,
            study_id=study_id,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def get_study_system_attrs(self, study_id: int) -> Dict[str, Any]:
        return self.client.sync(
            self.client.scheduler.optuna_get_study_system_attrs,
            study_id=study_id,
            storage_name=self.name,
        )

    async def _get_all_study_summaries(self) -> List[StudySummary]:
        serialized_summaries = await self.client.scheduler.optuna_get_all_study_summaries(
            storage_name=self.name
        )
        return [deserialize_studysummary(s) for s in serialized_summaries]

    @use_basestorage_doc
    def get_all_study_summaries(self) -> List[StudySummary]:
        return self.client.sync(self._get_all_study_summaries)

    # Basic trial manipulation

    @use_basestorage_doc
    def create_new_trial(self, study_id: int, template_trial: Optional[FrozenTrial] = None) -> int:
        return self.client.sync(
            self.client.scheduler.optuna_create_new_trial,
            study_id=study_id,
            template_trial=template_trial,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def set_trial_state(self, trial_id: int, state: TrialState) -> bool:
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_state,
            trial_id=trial_id,
            state=state.name,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
    ) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_param,
            trial_id=trial_id,
            param_name=param_name,
            param_value_internal=param_value_internal,
            distribution=distribution_to_json(distribution),
            storage_name=self.name,
        )

    @use_basestorage_doc
    def get_trial_number_from_id(self, trial_id: int) -> int:
        return self.client.sync(
            self.client.scheduler.optuna_get_trial_number_from_id,
            trial_id=trial_id,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def get_trial_param(self, trial_id: int, param_name: str) -> float:
        return self.client.sync(
            self.client.scheduler.optuna_get_trial_param,
            trial_id=trial_id,
            param_name=param_name,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def set_trial_value(self, trial_id: int, value: float) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_value,
            trial_id=trial_id,
            value=value,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_intermediate_value,
            trial_id=trial_id,
            step=step,
            intermediate_value=intermediate_value,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_user_attr,
            trial_id=trial_id,
            key=key,
            value=value,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_trial_system_attr,
            trial_id=trial_id,
            key=key,
            value=value,
            storage_name=self.name,
        )

    # Basic trial access
    async def _get_trial(self, trial_id: int) -> FrozenTrial:
        serialized_trial = await self.client.scheduler.optuna_get_trial(
            trial_id=trial_id, storage_name=self.name
        )
        return deserialize_frozentrial(serialized_trial)

    @use_basestorage_doc
    def get_trial(self, trial_id: int) -> FrozenTrial:
        return self.client.sync(self._get_trial, trial_id=trial_id)

    async def _get_all_trials(self, study_id: int, deepcopy: bool = True) -> List[FrozenTrial]:
        serialized_trials = await self.client.scheduler.optuna_get_all_trials(
            study_id=study_id,
            deepcopy=deepcopy,
            storage_name=self.name,
        )
        return [deserialize_frozentrial(t) for t in serialized_trials]

    @use_basestorage_doc
    def get_all_trials(self, study_id: int, deepcopy: bool = True) -> List[FrozenTrial]:
        return self.client.sync(
            self._get_all_trials,
            study_id=study_id,
            deepcopy=deepcopy,
        )

    @use_basestorage_doc
    def get_n_trials(self, study_id: int, state: Optional[TrialState] = None) -> int:
        return self.client.sync(
            self.client.scheduler.optuna_get_n_trials,
            study_id=study_id,
            state=state,
            storage_name=self.name,
        )

    @use_basestorage_doc
    def read_trials_from_remote_storage(self, study_id: int) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_read_trials_from_remote_storage,
            study_id=study_id,
            storage_name=self.name,
        )
