from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime
import json
import threading

from optuna.distributions import distribution_to_json
from optuna.distributions import json_to_distribution
from optuna.exceptions import DuplicatedStudyError
from optuna.storages import RDBStorage
from optuna.storages.grpc import _api_pb2
from optuna.storages.grpc import _api_pb2_grpc
from optuna.storages.grpc._grpc_imports import _imports
from optuna.study._study_direction import StudyDirection
from optuna.trial._frozen import FrozenTrial
from optuna.trial._state import TrialState


if _imports.is_successful():
    from optuna.storages.grpc._grpc_imports import grpc


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


class OptunaStorageProxyService(_api_pb2_grpc.StorageServiceServicer):
    def __init__(self, storage_url: str) -> None:
        self._backend = RDBStorage(storage_url)
        self._lock = threading.Lock()

    def CreateNewStudy(
        self,
        request: _api_pb2.CreateNewStudyRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.CreateNewStudyReply:
        directions = [
            StudyDirection.MINIMIZE if d == _api_pb2.MINIMIZE else StudyDirection.MAXIMIZE
            for d in request.directions
        ]
        study_name = request.study_name

        try:
            study_id = self._backend.create_new_study(directions=directions, study_name=study_name)
        except DuplicatedStudyError as e:
            context.abort(code=grpc.StatusCode.ALREADY_EXISTS, details=str(e))
        return _api_pb2.CreateNewStudyReply(study_id=study_id)

    def DeleteStudy(
        self,
        request: _api_pb2.DeleteStudyRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.DeleteStudyReply:
        study_id = request.study_id
        try:
            self._backend.delete_study(study_id)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        return _api_pb2.DeleteStudyReply()

    def SetStudyUserAttribute(
        self,
        request: _api_pb2.SetStudyUserAttributeRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.SetStudyUserAttributeReply:
        try:
            self._backend.set_study_user_attr(
                request.study_id, request.key, json.loads(request.value)
            )
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        return _api_pb2.SetStudyUserAttributeReply()

    def SetStudySystemAttribute(
        self,
        request: _api_pb2.SetStudySystemAttributeRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.SetStudySystemAttributeReply:
        try:
            self._backend.set_study_system_attr(
                request.study_id, request.key, json.loads(request.value)
            )
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        return _api_pb2.SetStudySystemAttributeReply()

    def GetStudyIdFromName(
        self,
        request: _api_pb2.GetStudyIdFromNameRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.GetStudyIdFromNameReply:
        try:
            study_id = self._backend.get_study_id_from_name(request.study_name)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        return _api_pb2.GetStudyIdFromNameReply(study_id=study_id)

    def GetStudyNameFromId(
        self,
        request: _api_pb2.GetStudyNameFromIdRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.GetStudyNameFromIdReply:
        study_id = request.study_id

        try:
            name = self._backend.get_study_name_from_id(study_id)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        assert name is not None
        return _api_pb2.GetStudyNameFromIdReply(study_name=name)

    def GetStudyDirections(
        self,
        request: _api_pb2.GetStudyDirectionsRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.GetStudyDirectionsReply:
        study_id = request.study_id

        try:
            directions = self._backend.get_study_directions(study_id)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))

        assert directions is not None
        return _api_pb2.GetStudyDirectionsReply(
            directions=[
                _api_pb2.MINIMIZE if d == StudyDirection.MINIMIZE else _api_pb2.MAXIMIZE
                for d in directions
            ]
        )

    def GetStudyUserAttributes(
        self,
        request: _api_pb2.GetStudyUserAttributesRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.GetStudyUserAttributesReply:
        try:
            attributes = self._backend.get_study_user_attrs(request.study_id)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        return _api_pb2.GetStudyUserAttributesReply(
            user_attributes={key: json.dumps(value) for key, value in attributes.items()}
        )

    def GetStudySystemAttributes(
        self,
        request: _api_pb2.GetStudySystemAttributesRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.GetStudySystemAttributesReply:
        try:
            attributes = self._backend.get_study_system_attrs(request.study_id)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        return _api_pb2.GetStudySystemAttributesReply(
            system_attributes={key: json.dumps(value) for key, value in attributes.items()}
        )

    def GetAllStudies(
        self,
        request: _api_pb2.GetAllStudiesRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.GetAllStudiesReply:
        studies = self._backend.get_all_studies()
        return _api_pb2.GetAllStudiesReply(
            frozen_studies=[
                _api_pb2.FrozenStudy(
                    study_id=study._study_id,
                    study_name=study.study_name,
                    directions=[
                        _api_pb2.MINIMIZE if d == StudyDirection.MINIMIZE else _api_pb2.MAXIMIZE
                        for d in study.directions
                    ],
                    user_attributes={
                        key: json.dumps(value) for key, value in study.user_attrs.items()
                    },
                    system_attributes={
                        key: json.dumps(value) for key, value in study.system_attrs.items()
                    },
                )
                for study in studies
            ]
        )

    def CreateNewTrial(
        self,
        request: _api_pb2.CreateNewTrialRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.CreateNewTrialReply:
        study_id = request.study_id

        template_trial = None
        if not request.template_trial_is_none:
            template_trial = _from_proto_frozen_trial(request.template_trial)

        try:
            frozen_trial = self._backend._create_new_trial(study_id, template_trial)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        trial_id = frozen_trial._trial_id

        return _api_pb2.CreateNewTrialReply(trial_id=trial_id)

    def SetTrialParameter(
        self,
        request: _api_pb2.SetTrialParameterRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.SetTrialParameterReply:
        trial_id = request.trial_id
        param_name = request.param_name
        param_value_internal = request.param_value_internal
        distribution = json_to_distribution(request.distribution)
        try:
            self._backend.set_trial_param(trial_id, param_name, param_value_internal, distribution)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        except RuntimeError as e:
            context.abort(code=grpc.StatusCode.FAILED_PRECONDITION, details=str(e))
        except ValueError as e:
            context.abort(code=grpc.StatusCode.INVALID_ARGUMENT, details=str(e))
        return _api_pb2.SetTrialParameterReply()

    def GetTrialIdFromStudyIdTrialNumber(
        self,
        request: _api_pb2.GetTrialIdFromStudyIdTrialNumberRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.GetTrialIdFromStudyIdTrialNumberReply:
        study_id = request.study_id
        trial_number = request.trial_number

        try:
            trial_id = self._backend.get_trial_id_from_study_id_trial_number(
                study_id, trial_number
            )
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        return _api_pb2.GetTrialIdFromStudyIdTrialNumberReply(trial_id=trial_id)

    def SetTrialStateValues(
        self,
        request: _api_pb2.SetTrialStateValuesRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.SetTrialStateValuesReply:
        trial_id = request.trial_id
        state = request.state
        values = request.values
        try:
            trial_updated = self._backend.set_trial_state_values(
                trial_id, _from_proto_trial_state(state), values
            )
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        except RuntimeError as e:
            context.abort(code=grpc.StatusCode.FAILED_PRECONDITION, details=str(e))
        return _api_pb2.SetTrialStateValuesReply(trial_updated=trial_updated)

    def SetTrialIntermediateValue(
        self,
        request: _api_pb2.SetTrialIntermediateValueRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.SetTrialIntermediateValueReply:
        trial_id = request.trial_id
        step = request.step
        intermediate_value = request.intermediate_value
        try:
            self._backend.set_trial_intermediate_value(trial_id, step, intermediate_value)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        except RuntimeError as e:
            context.abort(code=grpc.StatusCode.FAILED_PRECONDITION, details=str(e))
        return _api_pb2.SetTrialIntermediateValueReply()

    def SetTrialUserAttribute(
        self,
        request: _api_pb2.SetTrialUserAttributeRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.SetTrialUserAttributeReply:
        trial_id = request.trial_id
        key = request.key
        value = json.loads(request.value)
        try:
            self._backend.set_trial_user_attr(trial_id, key, value)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        except RuntimeError as e:
            context.abort(code=grpc.StatusCode.FAILED_PRECONDITION, details=str(e))
        return _api_pb2.SetTrialUserAttributeReply()

    def SetTrialSystemAttribute(
        self,
        request: _api_pb2.SetTrialSystemAttributeRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.SetTrialSystemAttributeReply:
        trial_id = request.trial_id
        key = request.key
        value = json.loads(request.value)
        try:
            self._backend.set_trial_system_attr(trial_id, key, value)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))
        except RuntimeError as e:
            context.abort(code=grpc.StatusCode.FAILED_PRECONDITION, details=str(e))
        return _api_pb2.SetTrialSystemAttributeReply()

    def GetTrial(
        self,
        request: _api_pb2.GetTrialRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.GetTrialReply:
        trial_id = request.trial_id
        try:
            trial = self._backend.get_trial(trial_id)
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))

        return _api_pb2.GetTrialReply(frozen_trial=_to_proto_frozen_trial(trial))

    def GetAllTrials(
        self,
        request: _api_pb2.GetAllTrialsRequest,
        context: grpc.ServicerContext,
    ) -> _api_pb2.GetAllTrialsReply:
        study_id = request.study_id
        deepcopy = request.deepcopy
        states = [_from_proto_trial_state(state) for state in request.states]
        try:
            trials = self._backend._get_trials(
                study_id,
                states=states,
                included_trial_ids=set(),
                trial_id_greater_than=-1,
            )
        except KeyError as e:
            context.abort(code=grpc.StatusCode.NOT_FOUND, details=str(e))

        trials = copy.deepcopy(trials) if deepcopy else trials
        return _api_pb2.GetAllTrialsReply(
            frozen_trials=[_to_proto_frozen_trial(trial) for trial in trials]
        )


def _to_proto_trial_state(state: TrialState) -> _api_pb2.TrialState.ValueType:
    if state == TrialState.RUNNING:
        return _api_pb2.RUNNING
    if state == TrialState.COMPLETE:
        return _api_pb2.COMPLETE
    if state == TrialState.PRUNED:
        return _api_pb2.PRUNED
    if state == TrialState.FAIL:
        return _api_pb2.FAIL
    if state == TrialState.WAITING:
        return _api_pb2.WAITING
    raise ValueError(f"Unknown TrialState: {state}")


def _from_proto_trial_state(state: _api_pb2.TrialState.ValueType) -> TrialState:
    if state == _api_pb2.RUNNING:
        return TrialState.RUNNING
    if state == _api_pb2.COMPLETE:
        return TrialState.COMPLETE
    if state == _api_pb2.PRUNED:
        return TrialState.PRUNED
    if state == _api_pb2.FAIL:
        return TrialState.FAIL
    if state == _api_pb2.WAITING:
        return TrialState.WAITING
    raise ValueError(f"Unknown _api_pb2.TrialState: {state}")


def _to_proto_frozen_trial(frozen_trial: FrozenTrial) -> _api_pb2.FrozenTrial:
    params = {}
    for key, value in frozen_trial.params.items():
        params[key] = frozen_trial.distributions[key].to_internal_repr(value)

    return _api_pb2.FrozenTrial(
        trial_id=frozen_trial._trial_id,
        number=frozen_trial.number,
        state=_to_proto_trial_state(frozen_trial.state),
        values=frozen_trial.values,
        datetime_start=(
            frozen_trial.datetime_start.strftime(DATETIME_FORMAT)
            if frozen_trial.datetime_start
            else ""
        ),
        datetime_complete=(
            frozen_trial.datetime_complete.strftime(DATETIME_FORMAT)
            if frozen_trial.datetime_complete
            else ""
        ),
        distributions={
            key: distribution_to_json(distribution)
            for key, distribution in frozen_trial.distributions.items()
        },
        params=params,
        user_attributes={key: json.dumps(value) for key, value in frozen_trial.user_attrs.items()},
        system_attributes={
            key: json.dumps(value) for key, value in frozen_trial.system_attrs.items()
        },
        intermediate_values={
            step: value for step, value in frozen_trial.intermediate_values.items()
        },
    )


def _from_proto_frozen_trial(frozen_trial: _api_pb2.FrozenTrial) -> FrozenTrial:
    datetime_start = (
        datetime.strptime(frozen_trial.datetime_start, DATETIME_FORMAT)
        if frozen_trial.datetime_start
        else None
    )
    datetime_complete = (
        datetime.strptime(frozen_trial.datetime_complete, DATETIME_FORMAT)
        if frozen_trial.datetime_complete
        else None
    )
    distributions = {
        key: json_to_distribution(value) for key, value in frozen_trial.distributions.items()
    }
    params = {}
    for key, value in frozen_trial.params.items():
        params[key] = distributions[key].to_external_repr(value)

    return FrozenTrial(
        trial_id=frozen_trial.trial_id,
        number=frozen_trial.number,
        state=_from_proto_trial_state(frozen_trial.state),
        value=None,
        values=frozen_trial.values if frozen_trial.values else None,
        datetime_start=datetime_start,
        datetime_complete=datetime_complete,
        params=params,
        distributions=distributions,
        user_attrs={key: json.loads(value) for key, value in frozen_trial.user_attributes.items()},
        system_attrs={
            key: json.loads(value) for key, value in frozen_trial.system_attributes.items()
        },
        intermediate_values={
            step: value for step, value in frozen_trial.intermediate_values.items()
        },
    )


def make_server(
    storage_url: str, host: str, port: int, thread_pool: ThreadPoolExecutor | None = None
) -> grpc.Server:
    server = grpc.server(thread_pool or ThreadPoolExecutor(max_workers=10))
    _api_pb2_grpc.add_StorageServiceServicer_to_server(
        OptunaStorageProxyService(storage_url), server
    )  # type: ignore
    server.add_insecure_port(f"{host}:{port}")
    return server


def run_server(
    storage_url: str, host: str, port: int, thread_pool: ThreadPoolExecutor | None = None
) -> None:
    """Run a gRPC server for the given storage URL, host, and port.

    Example:

        Run this server with the following way:

        .. code::

            from optuna.storages.grpc import run_server

            run_server("sqlite:///example.db", "localhost", 13000)

        Please refer to the client class :class:`~optuna.storages.grpc.GrpcStorageProxy` for
        the client usage.

    Args:
        storage_url: URL of the storage.
        host: Host to listen on.
        port: Port to listen on.
        thread_pool:
            Thread pool to use for the server. If :obj:`None`, a default thread pool
            with 10 workers will be used.
    """
    server = make_server(storage_url, host, port, thread_pool)
    server.start()
    print(f"Server started at {host}:{port}")
    print("Listening...")
    server.wait_for_termination()
