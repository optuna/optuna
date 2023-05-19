import asyncio
from datetime import datetime
from typing import Any
from typing import Container
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import uuid

import optuna
from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna._typing import JSONSerializable
from optuna.distributions import BaseDistribution
from optuna.distributions import distribution_to_json
from optuna.distributions import json_to_distribution
from optuna.storages import BaseStorage
from optuna.study import StudyDirection
from optuna.study._frozen import FrozenStudy
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


with try_import() as _imports:
    import distributed
    from distributed.protocol.pickle import dumps
    from distributed.protocol.pickle import loads
    from distributed.utils import thread_state  # type: ignore[attr-defined]
    from distributed.worker import get_client


def _serialize_frozentrial(trial: FrozenTrial) -> dict:
    data = trial.__dict__.copy()
    data["state"] = data["state"].name
    attrs = [a for a in data.keys() if a.startswith("_")]
    for attr in attrs:
        data[attr[1:]] = data.pop(attr)
    data["system_attrs"] = (
        dumps(data["system_attrs"])  # type: ignore[no-untyped-call]
        if data["system_attrs"]
        else {}
    )
    data["user_attrs"] = (
        dumps(data["user_attrs"]) if data["user_attrs"] else {}  # type: ignore[no-untyped-call]
    )
    data["distributions"] = {k: distribution_to_json(v) for k, v in data["distributions"].items()}
    if data["datetime_start"] is not None:
        data["datetime_start"] = data["datetime_start"].isoformat(timespec="microseconds")
    if data["datetime_complete"] is not None:
        data["datetime_complete"] = data["datetime_complete"].isoformat(timespec="microseconds")
    data["value"] = None
    return data


def _deserialize_frozentrial(data: dict) -> FrozenTrial:
    data["state"] = TrialState[data["state"]]
    data["distributions"] = {k: json_to_distribution(v) for k, v in data["distributions"].items()}
    if data["datetime_start"] is not None:
        data["datetime_start"] = datetime.fromisoformat(data["datetime_start"])
    if data["datetime_complete"] is not None:
        data["datetime_complete"] = datetime.fromisoformat(data["datetime_complete"])
    data["system_attrs"] = (
        loads(data["system_attrs"])  # type: ignore[no-untyped-call]
        if data["system_attrs"]
        else {}
    )
    data["user_attrs"] = (
        loads(data["user_attrs"]) if data["user_attrs"] else {}  # type: ignore[no-untyped-call]
    )
    return FrozenTrial(**data)


def _serialize_frozenstudy(study: FrozenStudy) -> dict:
    data = {
        "directions": [d.name for d in study._directions],
        "study_id": study._study_id,
        "study_name": study.study_name,
        "user_attrs": dumps(study.user_attrs)  # type: ignore[no-untyped-call]
        if study.user_attrs
        else {},
        "system_attrs": dumps(study.system_attrs)  # type: ignore[no-untyped-call]
        if study.system_attrs
        else {},
    }
    return data


def _deserialize_frozenstudy(data: dict) -> FrozenStudy:
    data["directions"] = [StudyDirection[d] for d in data["directions"]]
    data["direction"] = None
    data["system_attrs"] = (
        loads(data["system_attrs"])  # type: ignore[no-untyped-call]
        if data["system_attrs"]
        else {}
    )
    data["user_attrs"] = (
        loads(data["user_attrs"]) if data["user_attrs"] else {}  # type: ignore[no-untyped-call]
    )
    return FrozenStudy(**data)


class _OptunaSchedulerExtension:
    def __init__(self, scheduler: "distributed.Scheduler"):
        self.scheduler = scheduler
        self.storages: Dict[str, BaseStorage] = {}

        methods = [
            "create_new_study",
            "delete_study",
            "set_study_user_attr",
            "set_study_system_attr",
            "get_study_id_from_name",
            "get_study_name_from_id",
            "get_study_directions",
            "get_study_user_attrs",
            "get_study_system_attrs",
            "get_all_studies",
            "create_new_trial",
            "set_trial_param",
            "get_trial_id_from_study_id_trial_number",
            "get_trial_number_from_id",
            "get_trial_param",
            "set_trial_state_values",
            "set_trial_intermediate_value",
            "set_trial_user_attr",
            "set_trial_system_attr",
            "get_trial",
            "get_all_trials",
            "get_n_trials",
        ]
        handlers = {f"optuna_{method}": getattr(self, method) for method in methods}
        self.scheduler.handlers.update(handlers)

        self.scheduler.extensions["optuna"] = self

    def get_storage(self, name: str) -> BaseStorage:
        return self.storages[name]

    def create_new_study(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        directions: List[str],
        study_name: Optional[str] = None,
    ) -> int:
        return self.get_storage(storage_name).create_new_study(
            directions=[StudyDirection[direction] for direction in directions],
            study_name=study_name,
        )

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
            study_id=study_id, key=key, value=loads(value)  # type: ignore[no-untyped-call]
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
            value=loads(value),  # type: ignore[no-untyped-call]
        )

    def get_study_id_from_name(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_name: str,
    ) -> int:
        return self.get_storage(storage_name).get_study_id_from_name(study_name=study_name)

    def get_study_name_from_id(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
    ) -> str:
        return self.get_storage(storage_name).get_study_name_from_id(study_id=study_id)

    def get_study_directions(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
    ) -> List[str]:
        directions = self.get_storage(storage_name).get_study_directions(study_id=study_id)
        return [direction.name for direction in directions]

    def get_study_user_attrs(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
    ) -> Dict[str, Any]:
        return dumps(
            self.get_storage(storage_name).get_study_user_attrs(  # type: ignore[no-untyped-call]
                study_id=study_id
            )
        )

    def get_study_system_attrs(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
    ) -> Dict[str, Any]:
        return dumps(
            self.get_storage(storage_name).get_study_system_attrs(  # type: ignore[no-untyped-call]
                study_id=study_id
            )
        )

    def get_all_studies(self, comm: "distributed.comm.tcp.TCP", storage_name: str) -> List[dict]:
        studies = self.get_storage(storage_name).get_all_studies()
        return [_serialize_frozenstudy(s) for s in studies]

    def create_new_trial(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
        template_trial: Optional[dict] = None,
    ) -> int:
        deserialized_template_trial = None
        if template_trial is not None:
            deserialized_template_trial = _deserialize_frozentrial(template_trial)
        return self.get_storage(storage_name).create_new_trial(
            study_id=study_id,
            template_trial=deserialized_template_trial,
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

    def get_trial_id_from_study_id_trial_number(
        self, comm: "distributed.comm.tcp.TCP", storage_name: str, study_id: int, trial_number: int
    ) -> int:
        return self.get_storage(storage_name).get_trial_id_from_study_id_trial_number(
            study_id=study_id,
            trial_number=trial_number,
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

    def set_trial_state_values(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        trial_id: int,
        state: str,
        values: Optional[Sequence[float]] = None,
    ) -> bool:
        return self.get_storage(storage_name).set_trial_state_values(
            trial_id=trial_id,
            state=TrialState[state],
            values=values,
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
            value=loads(value),  # type: ignore[no-untyped-call]
        )

    def set_trial_system_attr(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        trial_id: int,
        key: str,
        value: JSONSerializable,
    ) -> None:
        return self.get_storage(storage_name).set_trial_system_attr(
            trial_id=trial_id,
            key=key,
            value=loads(value),  # type: ignore[no-untyped-call]
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
        states: Optional[Tuple[str, ...]] = None,
    ) -> List[dict]:
        deserialized_states = None
        if states is not None:
            deserialized_states = tuple(TrialState[s] for s in states)
        trials = self.get_storage(storage_name).get_all_trials(
            study_id=study_id,
            deepcopy=deepcopy,
            states=deserialized_states,
        )
        return [_serialize_frozentrial(t) for t in trials]

    def get_n_trials(
        self,
        comm: "distributed.comm.tcp.TCP",
        storage_name: str,
        study_id: int,
        state: Optional[Union[Tuple[str, ...], str]] = None,
    ) -> int:
        deserialized_state: Optional[Union[Tuple[TrialState, ...], TrialState]] = None
        if state is not None:
            if isinstance(state, str):
                deserialized_state = TrialState[state]
            else:
                deserialized_state = tuple(TrialState[s] for s in state)
        return self.get_storage(storage_name).get_n_trials(
            study_id=study_id,
            state=deserialized_state,
        )


def _register_with_scheduler(
    dask_scheduler: "distributed.Scheduler", storage: Union[None, str, BaseStorage], name: str
) -> None:
    if "optuna" not in dask_scheduler.extensions:
        ext = _OptunaSchedulerExtension(dask_scheduler)
    else:
        ext = dask_scheduler.extensions["optuna"]

    if name not in ext.storages:
        ext.storages[name] = optuna.storages.get_storage(storage)


@experimental_class("3.1.0")
class DaskStorage(BaseStorage):
    """Dask-compatible storage class.

    This storage class wraps a Optuna storage class (e.g. Optuna’s in-memory or sqlite storage)
    and is used to run optimization trials in parallel on a Dask cluster.
    The underlying Optuna storage object lives on the cluster’s scheduler and any method calls on
    the :obj:`DaskStorage` instance results in the same method being called on the underlying
    Optuna storage object.

    See `this example <https://github.com/optuna/optuna-examples/blob/master/
    dask/dask_simple.py>`_ or the following YouTube video
    for how to use :obj:`DaskStorage` to extend Optuna's in-memory storage class to run across
    multiple processes.

    .. raw:: html

       <iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/euT6_h7iIBA"
        frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media;
        gyroscope; picture-in-picture" allowfullscreen></iframe>
       <br>
       <br>

    Args:
        storage:
            Optuna storage url to use for underlying Optuna storage class to wrap
            (e.g. :obj:`None` for in-memory storage, ``sqlite:///example.db``
            for SQLite storage). Defaults to :obj:`None`.

        name:
            Unique identifier for the Dask storage class. Specifying a custom name can sometimes
            be useful for logging or debugging. If :obj:`None` is provided,
            a random name will be automatically generated.

        client:
            Dask ``Client`` to connect to. If not provided, will attempt to find an
            existing ``Client``.

        register:
            Whether or not to register this storage instance with the cluster scheduler.
            Most common usage of this storage class will not need to specify this argument.
            Defaults to ``True``.

    """

    def __init__(
        self,
        storage: Union[None, str, BaseStorage] = None,
        name: Optional[str] = None,
        client: Optional["distributed.Client"] = None,
        register: bool = True,
    ):
        _imports.check()
        self.name = name or f"dask-storage-{uuid.uuid4().hex}"
        self._client = client
        if register:
            if self.client.asynchronous or getattr(thread_state, "on_event_loop_thread", False):

                async def _register() -> DaskStorage:
                    await self.client.run_on_scheduler(  # type: ignore[no-untyped-call]
                        _register_with_scheduler, storage=storage, name=self.name
                    )
                    return self

                self._started = asyncio.ensure_future(_register())
            else:
                self.client.run_on_scheduler(  # type: ignore[no-untyped-call]
                    _register_with_scheduler, storage=storage, name=self.name
                )

    @property
    def client(self) -> "distributed.Client":
        if not self._client:
            self._client = get_client()
        return self._client

    def __await__(self) -> Generator[Any, None, "DaskStorage"]:
        if hasattr(self, "_started"):
            return self._started.__await__()
        else:

            async def _() -> DaskStorage:
                return self

            return _().__await__()

    def __reduce__(self) -> tuple:
        # We don't have a reference to underlying Optuna storage instance which lives
        # on the scheduler. This is okay since this DaskStorage instance has already been
        # registered with the scheduler, and ``storage`` is only ever needed during the
        # scheduler registration process. We use ``storage=None`` below by convention.
        return (DaskStorage, (None, self.name, None, False))

    def get_base_storage(self) -> BaseStorage:
        """Retrieve underlying Optuna storage instance from the scheduler.

        This is a convenience method to extract the Optuna storage instance stored on
        the Dask scheduler process to the local Python process.
        """

        def _get_base_storage(dask_scheduler: distributed.Scheduler, name: str) -> BaseStorage:
            return dask_scheduler.extensions["optuna"].storages[name]

        return self.client.run_on_scheduler(  # type: ignore[no-untyped-call]
            _get_base_storage, name=self.name
        )

    def create_new_study(
        self, directions: Sequence[StudyDirection], study_name: Optional[str] = None
    ) -> int:
        return self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_create_new_study,  # type: ignore[union-attr]
            storage_name=self.name,
            study_name=study_name,
            directions=[direction.name for direction in directions],
        )

    def delete_study(self, study_id: int) -> None:
        return self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_delete_study,  # type: ignore[union-attr]
            storage_name=self.name,
            study_id=study_id,
        )

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        return self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_set_study_user_attr,  # type: ignore[union-attr]
            storage_name=self.name,
            study_id=study_id,
            key=key,
            value=dumps(value),  # type: ignore[no-untyped-call]
        )

    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:
        return self.client.sync(
            self.client.scheduler.optuna_set_study_system_attr,  # type: ignore[union-attr]
            storage_name=self.name,
            study_id=study_id,
            key=key,
            value=dumps(value),  # type: ignore[no-untyped-call]
        )

    # Basic study access

    def get_study_id_from_name(self, study_name: str) -> int:
        return self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_get_study_id_from_name,  # type: ignore[union-attr]
            study_name=study_name,
            storage_name=self.name,
        )

    def get_study_name_from_id(self, study_id: int) -> str:
        return self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_get_study_name_from_id,  # type: ignore[union-attr]
            storage_name=self.name,
            study_id=study_id,
        )

    def get_study_directions(self, study_id: int) -> List[StudyDirection]:
        directions = self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_get_study_directions,  # type: ignore[union-attr]
            storage_name=self.name,
            study_id=study_id,
        )
        return [StudyDirection[direction] for direction in directions]

    def get_study_user_attrs(self, study_id: int) -> Dict[str, Any]:
        return loads(  # type: ignore[no-untyped-call]
            self.client.sync(  # type: ignore[no-untyped-call]
                self.client.scheduler.optuna_get_study_user_attrs,  # type: ignore[union-attr]
                storage_name=self.name,
                study_id=study_id,
            )
        )

    def get_study_system_attrs(self, study_id: int) -> Dict[str, Any]:
        return loads(  # type: ignore[no-untyped-call]
            self.client.sync(  # type: ignore[no-untyped-call]
                self.client.scheduler.optuna_get_study_system_attrs,  # type: ignore[union-attr]
                storage_name=self.name,
                study_id=study_id,
            )
        )

    def get_all_studies(self) -> List[FrozenStudy]:
        results = self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_get_all_studies,  # type: ignore[union-attr]
            storage_name=self.name,
        )
        return [_deserialize_frozenstudy(i) for i in results]

    # Basic trial manipulation

    def create_new_trial(self, study_id: int, template_trial: Optional[FrozenTrial] = None) -> int:
        serialized_template_trial = None
        if template_trial is not None:
            serialized_template_trial = _serialize_frozentrial(template_trial)
        return self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_create_new_trial,  # type: ignore[union-attr]
            storage_name=self.name,
            study_id=study_id,
            template_trial=serialized_template_trial,
        )

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
    ) -> None:
        return self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_set_trial_param,  # type: ignore[union-attr]
            storage_name=self.name,
            trial_id=trial_id,
            param_name=param_name,
            param_value_internal=param_value_internal,
            distribution=distribution_to_json(distribution),
        )

    def get_trial_id_from_study_id_trial_number(self, study_id: int, trial_number: int) -> int:
        return self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_get_trial_id_from_study_id_trial_number,  # type: ignore[union-attr]  # NOQA: E501
            storage_name=self.name,
            study_id=study_id,
            trial_number=trial_number,
        )

    def get_trial_number_from_id(self, trial_id: int) -> int:
        return self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_get_trial_number_from_id,  # type: ignore[union-attr]
            storage_name=self.name,
            trial_id=trial_id,
        )

    def get_trial_param(self, trial_id: int, param_name: str) -> float:
        return self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_get_trial_param,  # type: ignore[union-attr]
            storage_name=self.name,
            trial_id=trial_id,
            param_name=param_name,
        )

    def set_trial_state_values(
        self, trial_id: int, state: TrialState, values: Optional[Sequence[float]] = None
    ) -> bool:
        return self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_set_trial_state_values,  # type: ignore[union-attr]
            storage_name=self.name,
            trial_id=trial_id,
            state=state.name,
            values=values,
        )

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        return self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_set_trial_intermediate_value,  # type: ignore[union-attr]
            storage_name=self.name,
            trial_id=trial_id,
            step=step,
            intermediate_value=intermediate_value,
        )

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        return self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_set_trial_user_attr,  # type: ignore[union-attr]
            storage_name=self.name,
            trial_id=trial_id,
            key=key,
            value=dumps(value),  # type: ignore[no-untyped-call]
        )

    def set_trial_system_attr(self, trial_id: int, key: str, value: JSONSerializable) -> None:
        return self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_set_trial_system_attr,  # type: ignore[union-attr]
            storage_name=self.name,
            trial_id=trial_id,
            key=key,
            value=dumps(value),  # type: ignore[no-untyped-call]
        )

    # Basic trial access

    async def _get_trial(self, trial_id: int) -> FrozenTrial:
        serialized_trial = await self.client.scheduler.optuna_get_trial(  # type: ignore[union-attr]  # NOQA: E501
            trial_id=trial_id, storage_name=self.name
        )
        return _deserialize_frozentrial(serialized_trial)

    def get_trial(self, trial_id: int) -> FrozenTrial:
        return self.client.sync(  # type: ignore[no-untyped-call]
            self._get_trial, trial_id=trial_id
        )

    async def _get_all_trials(
        self, study_id: int, deepcopy: bool = True, states: Optional[Iterable[TrialState]] = None
    ) -> List[FrozenTrial]:
        serialized_states = None
        if states is not None:
            serialized_states = tuple(s.name for s in states)
        serialized_trials = await self.client.scheduler.optuna_get_all_trials(  # type: ignore[union-attr]  # NOQA: E501
            storage_name=self.name,
            study_id=study_id,
            deepcopy=deepcopy,
            states=serialized_states,
        )
        return [_deserialize_frozentrial(t) for t in serialized_trials]

    def get_all_trials(
        self, study_id: int, deepcopy: bool = True, states: Optional[Container[TrialState]] = None
    ) -> List[FrozenTrial]:
        return self.client.sync(  # type: ignore[no-untyped-call]
            self._get_all_trials,
            study_id=study_id,
            deepcopy=deepcopy,
            states=states,
        )

    def get_n_trials(
        self, study_id: int, state: Optional[Union[Tuple[TrialState, ...], TrialState]] = None
    ) -> int:
        serialized_state: Optional[Union[Tuple[str, ...], str]] = None
        if state is not None:
            if isinstance(state, TrialState):
                serialized_state = state.name
            else:
                serialized_state = tuple(s.name for s in state)
        return self.client.sync(  # type: ignore[no-untyped-call]
            self.client.scheduler.optuna_get_n_trials,  # type: ignore[union-attr]
            storage_name=self.name,
            study_id=study_id,
            state=serialized_state,
        )
