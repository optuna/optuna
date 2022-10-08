import copy
from datetime import datetime
from typing import Any
from typing import Container
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from uuid import uuid4

from google.auth.credentials import Credentials
from google.cloud import ndb

from optuna import distributions
from optuna.distributions import BaseDistribution
from optuna.distributions import distribution_to_json
from optuna.distributions import json_to_distribution
from optuna.storages import BaseStorage
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.study import StudyDirection
from optuna.study._frozen import FrozenStudy
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


class OptunaNDBTrial(ndb.Model):
    study_id = ndb.IntegerProperty(required=True)
    number = ndb.IntegerProperty(required=True, default=-1, indexed=True)
    state = ndb.IntegerProperty(
        required=True, choices=[s.value for s in TrialState], default=TrialState.RUNNING.value
    )
    values = ndb.FloatProperty(repeated=True)
    datetime_start = ndb.DateTimeProperty()
    datetime_complete = ndb.DateTimeProperty()
    params = ndb.JsonProperty()
    distributions = ndb.JsonProperty()
    user_attrs = ndb.JsonProperty(default={})
    system_attrs = ndb.JsonProperty(default={})
    intermediate_values = ndb.JsonProperty()


class OptunaNDBStudy(ndb.Model):
    study_name = ndb.StringProperty(required=True)
    directions = ndb.IntegerProperty(repeated=True, write_empty_list=True)
    param_distribution = ndb.JsonProperty(default={})
    user_attrs = ndb.JsonProperty(default={})
    system_attrs = ndb.JsonProperty(default={})
    trials = ndb.IntegerProperty(repeated=True, write_empty_list=True)


class DatastoreStorage(BaseStorage):
    """Storage class for Google Cloud Datastore backend.  The easiest way to use Google Cloud
    Datastore as your storage for Optuna is to use the environmental variable mechanism as
    explained here: https://cloud.google.com/docs/authentication/provide-credentials-adc
    Once your environment variable GOOGLE_APPLICATION_CREDENTIALS is properly setup,
    :class:`~optuna.storages.DatastoreStorage` will get your project name and credentials
    using this mechanism.


    Example:

        We create an :class:`~optuna.storages.DatastoreStorage` instance.

        .. code::

            import optuna


            def objective(trial):
                ...


            storage = optuna.storages.DatastoreStorage()

            study = optuna.create_study(storage=storage)
            study.optimize(objective)
    Args:
        gcp_project:
            The name of the Google Cloud project where your Datastore resource will be used
            with Optuna. If not provided, the project will be chosen from using the environment
            variable GOOGLE_APPLICATION_CREDENTIALS mechanism as explained here:
            https://cloud.google.com/docs/authentication/provide-credentials-adc

        namespace:
            The namespace withing Datastore to persist the OptunaNDBTrial and OptunaNDBStudy
            entities within Datastore. If not provide, the 'default' namespace in Datastore
            is used.

        gcp_credentials: Credentials
            If a Google Cloud Credentials object is provided, this object will be used to
            authenticate with Datastore. If not provided, the credential will be provided
            from the environment variable GOOGLE_APPLICATION_CREDENTIALS mechanism as
            explained here: https://cloud.google.com/docs/authentication/provide-credentials-adc

    .. note::
        If you use plan to use Datastore as a storage mechanism for optuna, a Google Cloud
        account is required along with a proper credentials mechanism. Please execute
        ``$ pip install -U google-cloud-ndb`` to install the required python libraries.
    """

    def __init__(
        self,
        gcp_project: Optional[str] = None,
        namespace: Optional[str] = None,
        gcp_credentials: Optional[Credentials] = None,
    ) -> None:
        self.ndb_client = ndb.Client(
            project=gcp_project, namespace=namespace, credentials=gcp_credentials
        )

    @staticmethod
    def ndb_study_to_optuna_study(ndb_study: OptunaNDBStudy) -> FrozenStudy:
        directions = [StudyDirection(d) for d in ndb_study.directions]
        frozen_study = FrozenStudy(
            study_id=ndb_study.key.id(),
            study_name=ndb_study.study_name,
            user_attrs=ndb_study.user_attrs,
            system_attrs=ndb_study.system_attrs,
            directions=directions,
            direction=None,
        )
        return frozen_study

    @staticmethod
    def optuna_trial_to_ndb_trial(trial: FrozenTrial, study_id: int) -> OptunaNDBTrial:

        # convert distribution objects to dicts for Datastore
        distributions_dict = dict()
        for param in trial.distributions:
            distributions_dict[param] = distribution_to_json(trial.distributions[param])

        ndb_trial = OptunaNDBTrial(
            study_id=study_id,
            number=trial.number,
            state=trial.state.value,
            params=trial.params,
            distributions=distributions_dict,
            user_attrs=trial.user_attrs,
            system_attrs=trial.system_attrs,
            intermediate_values=trial.intermediate_values,
            datetime_start=trial.datetime_start,
            datetime_complete=trial.datetime_complete,
        )

        if trial.values is not None and len(trial.values) > 0:
            ndb_trial.values = trial.values
        return ndb_trial

    @staticmethod
    def ndb_trial_to_optuna_trial(ndb_trial: OptunaNDBTrial) -> FrozenTrial:
        # convert distributions
        for param in ndb_trial.distributions:
            ndb_trial.distributions[param] = json_to_distribution(ndb_trial.distributions[param])
        frozen_trial = FrozenTrial(
            number=ndb_trial.number,
            state=TrialState(ndb_trial.state),
            value=None,
            values=ndb_trial.values,
            datetime_start=ndb_trial.datetime_start,
            datetime_complete=ndb_trial.datetime_complete,
            params=ndb_trial.params,
            distributions=ndb_trial.distributions,
            user_attrs=ndb_trial.user_attrs,
            system_attrs=ndb_trial.system_attrs,
            intermediate_values={int(k): v for k, v in ndb_trial.intermediate_values.items()},
            trial_id=ndb_trial.key.id(),
        )
        if frozen_trial.values is not None and len(frozen_trial.values) < 1:
            frozen_trial.values = None
        return frozen_trial

    @staticmethod
    def _create_running_trial() -> FrozenTrial:
        frozen_trial = FrozenTrial(
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
        frozen_trial.values = None
        return frozen_trial

    def create_new_study(self, study_name: Optional[str] = None) -> int:

        if study_name is None or len(study_name.strip()) < 1:
            study_uuid = str(uuid4())
            study_name = DEFAULT_STUDY_NAME_PREFIX + study_uuid

        with self.ndb_client.context():

            # check name pre-existence
            results = OptunaNDBStudy.query().filter(OptunaNDBStudy.study_name == study_name).get()
            if results is not None and (type(results) is OptunaNDBStudy or len(results) > 0):
                from optuna.exceptions import DuplicatedStudyError

                raise DuplicatedStudyError(f"Study with name {study_name} already exists.")

            study = OptunaNDBStudy(
                study_name=study_name, trials=[], directions=[StudyDirection.NOT_SET.value]
            )
            study.put()
            study_id = study.key.id()
            return study_id

    def get_study(self, study_id: int) -> OptunaNDBStudy:
        with self.ndb_client.context():
            study = OptunaNDBStudy.get_by_id(study_id)
            if study is None:
                raise KeyError(f"Study with id {study_id} not found in Datastore")
            return study

    def get_study_id_from_name(self, study_name: str) -> int:
        with self.ndb_client.context():
            study = OptunaNDBStudy.query(OptunaNDBStudy.study_name == study_name).get()
            if study is None:
                raise KeyError(f"Study with name {study_name} not found in Datastore")
            study_id = study.key.id()
            return study_id

    def set_study_directions(self, study_id: int, directions: Sequence[StudyDirection]) -> None:
        with self.ndb_client.context():
            study = OptunaNDBStudy.get_by_id(study_id)
            if study is None:
                raise KeyError(f"Study with id {study_id} not found in Datastore")

            if StudyDirection(study.directions[0]) != StudyDirection.NOT_SET and [
                StudyDirection(d) for d in study.directions
            ] != list(directions):
                raise ValueError(
                    f"Cannot overwrite study direction from {study.directions} to {directions}."
                )

            if len(directions) == 0:
                study.directions = [StudyDirection.NOT_SET.value]
            else:
                study.directions = [d.value for d in directions]

            study.put()

    def get_study_name_from_id(self, study_id: int) -> str:
        with self.ndb_client.context():
            study = OptunaNDBStudy.get_by_id(study_id)
            if study is None:
                raise KeyError(f"Study with id {study_id} not found in Datastore")

            return study.study_name

    def get_all_trials(
        self, study_id: int, deepcopy: bool = True, states: Optional[Container[TrialState]] = None
    ) -> List[FrozenTrial]:
        with self.ndb_client.context():
            study = OptunaNDBStudy.get_by_id(study_id)
            if study is None:
                raise KeyError(f"Study with id {study_id} not found in Datastore")

            study_trials = []
            for trial_id in study.trials:
                ndb_trial = OptunaNDBTrial.get_by_id(trial_id)
                if ndb_trial is None:
                    raise KeyError(f"Trial with id {trial_id} not found in Datastore")
                trial = self.ndb_trial_to_optuna_trial(ndb_trial)
                if states is None:
                    study_trials.append(trial)
                elif trial.state in states:
                    study_trials.append(trial)
            return study_trials

    def create_new_trial(self, study_id: int, template_trial: Optional[FrozenTrial] = None) -> int:
        with self.ndb_client.context():
            study = OptunaNDBStudy.get_by_id(study_id)
            if study is None:
                raise KeyError(f"Study with id {study_id} not found in Datastore")

            if template_trial is None:
                frozen_trial = self._create_running_trial()
            else:
                frozen_trial = copy.deepcopy(template_trial)

            frozen_trial.number = len(study.trials)
            ndb_trial = self.optuna_trial_to_ndb_trial(trial=frozen_trial, study_id=study_id)
            ndb_trial.put()
            trial_id = ndb_trial.key.id()

            study.trials.append(trial_id)
            study.put()
            return trial_id

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: BaseDistribution,
    ) -> None:
        with self.ndb_client.context():
            trial = OptunaNDBTrial.get_by_id(trial_id)
            if trial is None:
                raise KeyError(f"Trial with id {trial_id} not found in Datastore")

            self.check_trial_is_updatable(trial_id, TrialState(trial.state))

            study = OptunaNDBStudy.get_by_id(trial.study_id)

            if param_name in study.param_distribution:
                distributions.check_distribution_compatibility(
                    json_to_distribution(study.param_distribution[param_name]), distribution
                )

            trial.params[param_name] = distribution.to_external_repr(param_value_internal)
            trial.distributions[param_name] = distribution_to_json(distribution)

            study.param_distribution[param_name] = distribution_to_json(distribution)

            trial.put()
            study.put()

    def get_trial(self, trial_id: int) -> FrozenTrial:
        with self.ndb_client.context():
            trial = OptunaNDBTrial.get_by_id(trial_id)
            if trial is None:
                raise KeyError(f"Trial with id {trial_id} not found in Datastore")

            frozen_trial = self.ndb_trial_to_optuna_trial(trial)
            return frozen_trial

    def get_study_directions(self, study_id: int) -> List[StudyDirection]:
        with self.ndb_client.context():
            study = OptunaNDBStudy.get_by_id(study_id)
            if study is None:
                raise KeyError(f"Study with id {study_id} not found in Datastore")

            return [StudyDirection(d) for d in study.directions]

    def set_trial_state_values(
        self, trial_id: int, state: TrialState, values: Optional[Sequence[float]] = None
    ) -> bool:
        with self.ndb_client.context():
            trial = OptunaNDBTrial.get_by_id(trial_id)
            if trial is None:
                raise KeyError(f"Trial with id {trial_id} not found in Datastore")

            self.check_trial_is_updatable(trial_id, TrialState(trial.state))

            if state == TrialState.RUNNING and TrialState(trial.state) != TrialState.WAITING:
                return False

            trial.state = state.value

            if values is None and (trial.values is None or len(trial.values) < 1):
                trial.values = []
            elif type(values) in [float, int]:
                trial.values = [values]
            else:
                trial.values = values

            if state == TrialState.RUNNING:
                trial.datetime_start = datetime.now()

            if state.is_finished():
                trial.datetime_complete = datetime.now()

            trial.put()

            return True

    def get_all_studies(self) -> List[FrozenStudy]:
        with self.ndb_client.context():
            all_studies = []
            for s in OptunaNDBStudy.query().iter():
                all_studies.append(self.ndb_study_to_optuna_study(ndb_study=s))
            return all_studies

    def delete_study(self, study_id: int) -> None:
        with self.ndb_client.context():
            study = OptunaNDBStudy.get_by_id(study_id)
            if study is None:
                raise KeyError(f"Study with id {study_id} not found in Datastore")

            for trial_id in study.trials:
                trial_key = ndb.Key(OptunaNDBTrial, trial_id)
                if trial_key.get() is None:
                    raise KeyError(f"Study ID {trial_id} not found in Datastore")
                trial_key.delete()

            study_key = ndb.Key(OptunaNDBStudy, study_id)
            if study_key.get() is None:
                raise KeyError(f"Study ID {study_id} not found in Datastore")
            study_key.delete()

    def delete_trial(self, trial_id: int) -> None:
        with self.ndb_client.context():
            key = ndb.Key(OptunaNDBTrial, trial_id)
            if key.get() is None:
                raise KeyError(f"Study ID {trial_id} not found in Datastore")
            key.delete()

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        with self.ndb_client.context():
            trial = OptunaNDBTrial.get_by_id(trial_id)
            if trial is None:
                raise KeyError(f"Trial with id {trial_id} not found in Datastore")
            if trial.state == TrialState.COMPLETE.value:
                raise RuntimeError(f"Trial {trial_id} state is {trial.state}")
            trial.intermediate_values[step] = intermediate_value
            trial.put()

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:
        with self.ndb_client.context():
            study = OptunaNDBStudy.get_by_id(study_id)
            if study is None:
                raise KeyError(f"Study with id {study_id} not found in Datastore")
            study.user_attrs[key] = value
            study.put()

    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:
        with self.ndb_client.context():
            study = OptunaNDBStudy.get_by_id(study_id)
            if study is None:
                raise KeyError(f"Study with id {study_id} not found in Datastore")
            study.system_attrs[key] = value
            study.put()

    def get_study_user_attrs(self, study_id: int) -> Dict[str, Any]:
        with self.ndb_client.context():
            study = OptunaNDBStudy.get_by_id(study_id)
            if study is None:
                raise KeyError(f"Study with id {study_id} not found in Datastore")
            return study.user_attrs

    def get_study_system_attrs(self, study_id: int) -> Dict[str, Any]:
        with self.ndb_client.context():
            study = OptunaNDBStudy.get_by_id(study_id)
            if study is None:
                raise KeyError(f"Study with id {study_id} not found in Datastore")
            return study.system_attrs

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        with self.ndb_client.context():
            trial = OptunaNDBTrial.get_by_id(trial_id)
            if trial is None:
                raise KeyError(f"Trial with id {trial_id} not found in Datastore")
            if trial.state == TrialState.COMPLETE.value:
                raise RuntimeError(f"Trial {trial_id} state is {trial.state}")

            trial.user_attrs[key] = value
            trial.put()

    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:
        with self.ndb_client.context():
            trial = OptunaNDBTrial.get_by_id(trial_id)
            if trial is None:
                raise KeyError(f"Trial with id {trial_id} not found in Datastore")
            if trial.state == TrialState.COMPLETE.value:
                raise RuntimeError(f"Trial {trial_id} state is {trial.state}")

            trial.system_attrs[key] = value
            trial.put()

    def get_trial_param(self, trial_id: int, param_name: str) -> float:
        with self.ndb_client.context():
            trial = OptunaNDBTrial.get_by_id(trial_id)
            if trial is None:
                raise KeyError(f"Trial with id {trial_id} not found in Datastore")

            distribution_json = trial.distributions[param_name]
            distribution = json_to_distribution(distribution_json)
            trial_param = distribution.to_internal_repr(trial.params[param_name])
            return trial_param

    def get_trial_id_from_study_id_trial_number(self, study_id: int, trial_number: int) -> int:
        with self.ndb_client.context():
            if study_id < 1:
                raise KeyError(
                    f"""Study ID {study_id} not found in Datastore.
                    Google Datastore ids must non-zero."""
                )
            study = OptunaNDBStudy.get_by_id(study_id)
            if study is None:
                raise KeyError(f"Study with id {study_id} not found in Datastore")

            if len(study.trials) < 1:
                raise KeyError(
                    f"""Trial number {trial_number} does not exists
                     in study with study_id {study_id}."""
                )

            for trial_id in study.trials:
                trial = OptunaNDBTrial.get_by_id(trial_id)
                if trial.number == trial_number:
                    return trial_id

            raise KeyError(
                f"""Trial number {trial_number} does not exists
                                 in study with study_id {study_id}."""
            )
