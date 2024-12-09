from __future__ import annotations

from collections.abc import Container
from collections.abc import Iterable
from collections.abc import Sequence
import json
from typing import Any
import uuid

from optuna.distributions import BaseDistribution
from optuna.distributions import distribution_to_json
from optuna.exceptions import DuplicatedStudyError
from optuna.storages._base import BaseStorage
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.storages.grpc import _api_pb2
from optuna.storages.grpc._api_pb2_grpc import StorageServiceStub
from optuna.storages.grpc._grpc_imports import _imports
from optuna.storages.grpc._server import _from_proto_frozen_trial
from optuna.storages.grpc._server import _to_proto_frozen_trial
from optuna.storages.grpc._server import _to_proto_trial_state
from optuna.study._frozen import FrozenStudy
from optuna.study._study_direction import StudyDirection
from optuna.trial._frozen import FrozenTrial
from optuna.trial._state import TrialState


if _imports.is_successful():
    from optuna.storages.grpc._grpc_imports import grpc


class GrpcStorageProxy(BaseStorage):
    """gRPC client for :func:`~optuna.storages.grpc.run_server`.

    Example:

        This is a simple example of using :class:`~optuna.storages.grpc.GrpcStorageProxy` with
        :func:`~optuna.storages.grpc.run_server`.

        .. testcode::

            import optuna
            from optuna.storages.grpc import GrpcStorageProxy

            storage = GrpcStorageProxy(host="localhost", port=13000)
            study = optuna.create_study(storage=storage)

        Please refer to the example in :func:`~optuna.storages.grpc.run_server` for the server side
        code.

    Args:
        host: The host of the gRPC server.
        port: The port of the gRPC server.

    """

    def __init__(self, *, host: str = "localhost", port: int = 13000) -> None:
        self._stub = StorageServiceStub(grpc.insecure_channel(f"{host}:{port}"))  # type: ignore
        self._host = host
        self._port = port

    def __getstate__(self) -> dict[Any, Any]:
        state = self.__dict__.copy()
        del state["_stub"]
        return state

    def __setstate__(self, state: dict[Any, Any]) -> None:
        self.__dict__.update(state)
        self._stub = StorageServiceStub(
            grpc.insecure_channel(f"{self._host}:{self._port}")
        )  # type: ignore

    def create_new_study(
        self, directions: Sequence[StudyDirection], study_name: str | None = None
    ) -> int:
        request = _api_pb2.CreateNewStudyRequest(
            directions=[
                _api_pb2.MINIMIZE if d == StudyDirection.MINIMIZE else _api_pb2.MAXIMIZE
                for d in directions
            ],
            study_name=study_name
            or DEFAULT_STUDY_NAME_PREFIX
            + str(uuid.uuid4()),  # TODO(HideakiImamura): Check if this is unique.
        )
        try:
            response = self._stub.CreateNewStudy(request)
        except grpc.RpcError as e:
            raise DuplicatedStudyError from e
        return response.study_id

    def delete_study(self, study_id: int) -> None:
        request = _api_pb2.DeleteStudyRequest(study_id=study_id)
        try:
            self._stub.DeleteStudy(request)
        except grpc.RpcError as e:
            raise KeyError from e

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        request = _api_pb2.SetStudyUserAttributeRequest(
            study_id=study_id, key=key, value=json.dumps(value)
        )
        try:
            self._stub.SetStudyUserAttribute(request)
        except grpc.RpcError as e:
            raise KeyError from e

    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:
        request = _api_pb2.SetStudySystemAttributeRequest(
            study_id=study_id, key=key, value=json.dumps(value)
        )
        try:
            self._stub.SetStudySystemAttribute(request)
        except grpc.RpcError as e:
            raise KeyError from e

    def get_study_id_from_name(self, study_name: str) -> int:
        request = _api_pb2.GetStudyIdFromNameRequest(study_name=study_name)
        try:
            response = self._stub.GetStudyIdFromName(request)
        except grpc.RpcError as e:
            raise KeyError from e
        return response.study_id

    def get_study_name_from_id(self, study_id: int) -> str:
        request = _api_pb2.GetStudyNameFromIdRequest(study_id=study_id)
        try:
            response = self._stub.GetStudyNameFromId(request)
        except grpc.RpcError as e:
            raise KeyError from e
        return response.study_name

    def get_study_directions(self, study_id: int) -> list[StudyDirection]:
        request = _api_pb2.GetStudyDirectionsRequest(study_id=study_id)
        try:
            response = self._stub.GetStudyDirections(request)
        except grpc.RpcError as e:
            raise KeyError from e
        return [
            StudyDirection.MINIMIZE if d == _api_pb2.MINIMIZE else StudyDirection.MAXIMIZE
            for d in response.directions
        ]

    def get_study_user_attrs(self, study_id: int) -> dict[str, Any]:
        request = _api_pb2.GetStudyUserAttributesRequest(study_id=study_id)
        try:
            response = self._stub.GetStudyUserAttributes(request)
        except grpc.RpcError as e:
            raise KeyError from e
        return {key: json.loads(value) for key, value in response.user_attributes.items()}

    def get_study_system_attrs(self, study_id: int) -> dict[str, Any]:
        request = _api_pb2.GetStudySystemAttributesRequest(study_id=study_id)
        try:
            response = self._stub.GetStudySystemAttributes(request)
        except grpc.RpcError as e:
            raise KeyError from e
        return {key: json.loads(value) for key, value in response.system_attributes.items()}

    def get_all_studies(self) -> list[FrozenStudy]:
        request = _api_pb2.GetAllStudiesRequest()
        response = self._stub.GetAllStudies(request)
        return [
            FrozenStudy(
                study_id=study.study_id,
                study_name=study.study_name,
                direction=None,
                directions=[
                    StudyDirection.MINIMIZE if d == _api_pb2.MINIMIZE else StudyDirection.MAXIMIZE
                    for d in study.directions
                ],
                user_attrs={
                    key: json.loads(value) for key, value in study.user_attributes.items()
                },
                system_attrs={
                    key: json.loads(value) for key, value in study.system_attributes.items()
                },
            )
            for study in response.frozen_studies
        ]

    def create_new_trial(self, study_id: int, template_trial: FrozenTrial | None = None) -> int:
        if template_trial is None:
            request = _api_pb2.CreateNewTrialRequest(
                study_id=study_id, template_trial_is_none=True
            )
        else:
            request = _api_pb2.CreateNewTrialRequest(
                study_id=study_id,
                template_trial=_to_proto_frozen_trial(template_trial),
                template_trial_is_none=False,
            )
        try:
            response = self._stub.CreateNewTrial(request)
        except grpc.RpcError as e:
            raise KeyError from e
        return response.trial_id

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
    ) -> None:
        request = _api_pb2.SetTrialParameterRequest(
            trial_id=trial_id,
            param_name=param_name,
            param_value_internal=param_value_internal,
            distribution=distribution_to_json(distribution),
        )
        try:
            self._stub.SetTrialParameter(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError from e
            elif e.code() == grpc.StatusCode.INVALID_ARGUMENT:
                raise ValueError from e

    def set_trial_state_values(
        self, trial_id: int, state: TrialState, values: Sequence[float] | None = None
    ) -> bool:
        request = _api_pb2.SetTrialStateValuesRequest(
            trial_id=trial_id,
            state=_to_proto_trial_state(state),
            values=values,
        )
        try:
            response = self._stub.SetTrialStateValues(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError from e

        return response.trial_updated

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        request = _api_pb2.SetTrialIntermediateValueRequest(
            trial_id=trial_id, step=step, intermediate_value=intermediate_value
        )
        try:
            self._stub.SetTrialIntermediateValue(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError from e

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        request = _api_pb2.SetTrialUserAttributeRequest(
            trial_id=trial_id, key=key, value=json.dumps(value)
        )
        try:
            self._stub.SetTrialUserAttribute(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError from e

    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:
        request = _api_pb2.SetTrialSystemAttributeRequest(
            trial_id=trial_id, key=key, value=json.dumps(value)
        )
        try:
            self._stub.SetTrialSystemAttribute(request)
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                raise KeyError from e
            elif e.code() == grpc.StatusCode.FAILED_PRECONDITION:
                raise RuntimeError from e

    def get_trial_id_from_study_id_trial_number(self, study_id: int, trial_number: int) -> int:
        request = _api_pb2.GetTrialIdFromStudyIdTrialNumberRequest(
            study_id=study_id, trial_number=trial_number
        )
        try:
            response = self._stub.GetTrialIdFromStudyIdTrialNumber(request)
        except grpc.RpcError as e:
            raise KeyError from e
        return response.trial_id

    def get_trial(self, trial_id: int) -> FrozenTrial:
        request = _api_pb2.GetTrialRequest(trial_id=trial_id)
        try:
            response = self._stub.GetTrial(request)
        except grpc.RpcError as e:
            raise KeyError from e
        return _from_proto_frozen_trial(response.frozen_trial)

    def get_all_trials(
        self,
        study_id: int,
        deepcopy: bool = True,
        states: Container[TrialState] | None = None,
    ) -> list[FrozenTrial]:
        if states is None:
            states = [
                TrialState.RUNNING,
                TrialState.COMPLETE,
                TrialState.PRUNED,
                TrialState.FAIL,
                TrialState.WAITING,
            ]
        assert isinstance(states, Iterable)
        request = _api_pb2.GetAllTrialsRequest(
            study_id=study_id,
            deepcopy=deepcopy,
            states=[_to_proto_trial_state(state) for state in states],
        )
        try:
            response = self._stub.GetAllTrials(request)
        except grpc.RpcError as e:
            raise KeyError from e
        return [_from_proto_frozen_trial(trial) for trial in response.frozen_trials]
