from collections import defaultdict
from datetime import datetime
import json
from sqlalchemy.engine import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import orm
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA
import uuid

from pfnopt import distributions
from pfnopt import logging
from pfnopt.storages.base import BaseStorage
from pfnopt.storages.base import SYSTEM_ATTRS_KEY
from pfnopt.storages.rdb import models
from pfnopt import structs
from pfnopt import version


class RDBStorage(BaseStorage):

    def __init__(self, url, connect_args=None):
        # type: (str, Optional[Dict[str, Any]]) -> None

        connect_args = connect_args or {}

        url = self._fill_storage_url_template(url)

        try:
            self.engine = create_engine(url, connect_args=connect_args)
        except ImportError as e:
            raise ImportError(
                'Failed to import DB access module for the specified storage URL. '
                'Please install appropriate one. (The actual import error is: ' + str(e) + '.)')

        self.scoped_session = orm.scoped_session(orm.sessionmaker(bind=self.engine))
        models.BaseModel.metadata.create_all(self.engine)
        self._check_table_schema_compatibility()
        self.logger = logging.get_logger(__name__)

    def create_new_study_id(self):
        # type: () -> int

        session = self.scoped_session()

        while True:
            study_uuid = str(uuid.uuid4())
            study = models.StudyModel.find_by_uuid(study_uuid, session)
            if study is None:
                break

        study = models.StudyModel(study_uuid=study_uuid, task=structs.StudyTask.NOT_SET)
        session.add(study)
        self._commit(session)

        # Set system attribute key and empty value.
        self.set_study_user_attr(study.study_id, SYSTEM_ATTRS_KEY, {})

        self.logger.info('A new study created with UUID: {}'.format(study.study_uuid))

        return study.study_id

    # TODO(sano): Prevent simultaneous setting of different tasks by multiple threads/processes.
    def set_study_task(self, study_id, task):
        # type: (int, structs.StudyTask) -> None

        session = self.scoped_session()

        study = models.StudyModel.find_or_raise_by_id(study_id, session)

        if study.task != structs.StudyTask.NOT_SET and study.task != task:
            raise ValueError(
                'Cannot overwrite study task from {} to {}.'.format(study.task, task))

        study.task = task

        self._commit(session)

    def set_study_user_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        session = self.scoped_session()

        study = models.StudyModel.find_or_raise_by_id(study_id, session)
        attribute = models.StudyUserAttributeModel.find_by_study_and_key(study, key, session)
        if attribute is None:
            attribute = models.StudyUserAttributeModel(
                study_id=study_id, key=key, value_json=json.dumps(value))
            session.add(attribute)
        else:
            attribute.value_json = json.dumps(value)

        self._commit(session)

    def get_study_id_from_uuid(self, study_uuid):
        # type: (str) -> int

        session = self.scoped_session()

        study = models.StudyModel.find_or_raise_by_uuid(study_uuid, session)

        return study.study_id

    def get_study_uuid_from_id(self, study_id):
        # type: (int) -> str

        session = self.scoped_session()

        study = models.StudyModel.find_or_raise_by_id(study_id, session)

        return study.study_uuid

    def get_study_task(self, study_id):
        # type: (int) -> structs.StudyTask

        session = self.scoped_session()

        study = models.StudyModel.find_or_raise_by_id(study_id, session)

        return study.task

    def get_study_user_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        session = self.scoped_session()

        attributes = models.StudyUserAttributeModel.where_study_id(study_id, session)

        return {attr.key: json.loads(attr.value_json) for attr in attributes}

    # TODO(sano): Optimize this method to reduce the number of queries.
    def get_all_study_summaries(self):
        # type: () -> List[structs.StudySummary]

        session = self.scoped_session()

        study_models = models.StudyModel.all(session)
        trial_models = models.TrialModel.all(session)
        param_models = models.TrialParamModel.all(session)
        value_models = models.TrialValueModel.all(session)

        study_sumarries = []
        for study_model in study_models:
            # Filter model objects by study.
            study_trial_models = [t for t in trial_models if t.study_id == study_model.study_id]

            # Get best trial.
            completed_trial_models = [t for t in study_trial_models
                                      if t.state is structs.TrialState.COMPLETE]
            best_trial = None
            if len(completed_trial_models) > 0:
                # TODO(sano): Deal with maximize task.
                best_trial_model = min(completed_trial_models, key=lambda t: t.value)

                best_param_models = [p for p in param_models
                                     if p.trial_id == best_trial_model.trial_id]
                best_value_models = [v for v in value_models
                                     if v.trial_id == best_trial_model.trial_id]

                # Merge model objects related to the best trial.
                best_trial = self._merge_trials_orm(
                    [best_trial_model], best_param_models, best_value_models)[0]

            # Find datetime_start.
            datetime_start = None
            if len(study_trial_models) > 0:
                datetime_start = min([t.datetime_start for t in study_trial_models])

            # Consolidate StudySummary.
            study_sumarries.append(structs.StudySummary(
                study_id=study_model.study_id,
                study_uuid=study_model.study_uuid,
                task=self.get_study_task(study_model.study_id),
                best_trial=best_trial,
                user_attrs=self.get_study_user_attrs(study_model.study_id),
                n_trials=len(study_trial_models),
                datetime_start=datetime_start
            ))

        return study_sumarries

    def create_new_trial_id(self, study_id):
        # type: (int) -> int

        session = self.scoped_session()

        trial = models.TrialModel(
            study_id=study_id,
            state=structs.TrialState.RUNNING,
            user_attributes_json=json.dumps({SYSTEM_ATTRS_KEY: {}})
        )

        session.add(trial)
        self._commit(session)

        return trial.trial_id

    def set_trial_state(self, trial_id, state):
        # type: (int, structs.TrialState) -> None

        session = self.scoped_session()

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        trial.state = state
        if state.is_finished():
            trial.datetime_complete = datetime.now()

        self._commit(session)

    def set_trial_param(self, trial_id, param_name, param_value_internal, distribution):
        # type: (int, str, float, distributions.BaseDistribution) -> bool

        session = self.scoped_session()

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        trial_param = \
            models.TrialParamModel.find_by_trial_and_param_name(trial, param_name, session)

        if trial_param is not None:
            # Raise error in case distribution is incompatible.
            distributions.check_distribution_compatibility(
                distributions.json_to_distribution(trial_param.distribution_json),
                distribution
            )

            # Return False when distribution is compatible but parameter has already been set.
            return False

        param = models.TrialParamModel(
            trial_id=trial_id,
            param_name=param_name,
            param_value=param_value_internal,
            distribution_json=distributions.distribution_to_json(distribution)
        )

        param.check_and_add(session)
        commit_success = self._commit_or_rollback_on_integrity_error(session)

        return commit_success

    def get_trial_param(self, trial_id, param_name):
        # type: (int, str) -> float

        session = self.scoped_session()

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        trial_param = models.TrialParamModel.find_or_raise_by_trial_and_param_name(
            trial, param_name, session)

        return trial_param.param_value

    def set_trial_value(self, trial_id, value):
        # type: (int, float) -> None

        session = self.scoped_session()

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        trial.value = value

        self._commit(session)

    def set_trial_intermediate_value(self, trial_id, step, intermediate_value):
        # type: (int, int, float) -> bool

        session = self.scoped_session()

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        trial_value = models.TrialValueModel.find_by_trial_and_step(trial, step, session)
        if trial_value is not None:
            return False

        trial_value = models.TrialValueModel(
            trial_id=trial_id,
            step=step,
            value=intermediate_value
        )

        session.add(trial_value)
        commit_success = self._commit_or_rollback_on_integrity_error(session)

        return commit_success

    def set_trial_user_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        session = self.scoped_session()

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        loaded_json = json.loads(trial.user_attributes_json)
        loaded_json[key] = value
        trial.user_attributes_json = json.dumps(loaded_json)

        self._commit(session)

    def get_trial(self, trial_id):
        # type: (int) -> structs.FrozenTrial

        session = self.scoped_session()

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        params = models.TrialParamModel.where_trial(trial, session)
        values = models.TrialValueModel.where_trial(trial, session)

        return self._merge_trials_orm([trial], params, values)[0]

    def get_all_trials(self, study_id):
        # type: (int) -> List[structs.FrozenTrial]

        session = self.scoped_session()

        study = models.StudyModel.find_or_raise_by_id(study_id, session)
        trials = models.TrialModel.where_study(study, session)
        params = models.TrialParamModel.where_study(study, session)
        values = models.TrialValueModel.where_study(study, session)

        return self._merge_trials_orm(trials, params, values)

    def get_n_trials(self, study_id, state=None):
        # type: (int, Optional[structs.TrialState]) -> int

        session = self.scoped_session()
        study = models.StudyModel.find_or_raise_by_id(study_id, session)
        return models.TrialModel.count(session, study, state)

    @staticmethod
    def _merge_trials_orm(
            trials,  # type: List[models.TrialModel]
            trial_params,   # type: List[models.TrialParamModel]
            trial_intermediate_values  # type: List[models.TrialValueModel]
    ):
        # type: (...) -> List[structs.FrozenTrial]

        id_to_trial = {}
        for trial in trials:
            id_to_trial[trial.trial_id] = trial

        id_to_params = defaultdict(list)  # type: Dict[int, List[models.TrialParamModel]]
        for param in trial_params:
            id_to_params[param.trial_id].append(param)

        id_to_values = defaultdict(list)  # type: Dict[int, List[models.TrialValueModel]]
        for value in trial_intermediate_values:
            id_to_values[value.trial_id].append(value)

        result = []
        for trial_id, trial in id_to_trial.items():
            params = {}
            params_in_internal_repr = {}
            for param in id_to_params[trial_id]:
                distribution = distributions.json_to_distribution(param.distribution_json)
                params[param.param_name] = distribution.to_external_repr(param.param_value)
                params_in_internal_repr[param.param_name] = param.param_value

            intermediate_values = {}
            for value in id_to_values[trial_id]:
                intermediate_values[value.step] = value.value

            result.append(structs.FrozenTrial(
                trial_id=trial_id,
                state=trial.state,
                params=params,
                user_attrs=json.loads(trial.user_attributes_json),
                value=trial.value,
                intermediate_values=intermediate_values,
                params_in_internal_repr=params_in_internal_repr,
                datetime_start=trial.datetime_start,
                datetime_complete=trial.datetime_complete
            ))

        return result

    def _check_table_schema_compatibility(self):
        # type: () -> None

        session = self.scoped_session()

        version_info = models.VersionInfoModel.find(session)
        if version_info is not None:
            if version_info.schema_version != models.SCHEMA_VERSION:
                raise RuntimeError(
                    'The runtime pfnopt version {} is no longer compatible with the table schema '
                    '(set up by pfnopt {}).'.format(
                        version.__version__, version_info.library_version))
            return

        version_info = models.VersionInfoModel(
            schema_version=models.SCHEMA_VERSION,
            library_version=version.__version__
        )

        session.add(version_info)
        self._commit_or_rollback_on_integrity_error(session)

    @staticmethod
    def _fill_storage_url_template(template):
        # type: (str) -> str

        return template.format(SCHEMA_VERSION=models.SCHEMA_VERSION)

    def _commit_or_rollback_on_integrity_error(self, session):
        # type: (orm.Session) -> bool

        try:
            session.commit()
        except IntegrityError as e:
            self.logger.debug(
                'Ignoring {}. This happens due to a timing issue among threads/processes/nodes. '
                'Another one might have committed a record with the same key(s).'.format(repr(e)))
            session.rollback()
            return False

        return True

    def _commit(self, session):
        # type: (orm.Session) -> None

        try:
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            message = \
                'Raising {}. This happens due to a timing issue among threads/processes/nodes. ' \
                'Another one might have committed an invalid record. ' \
                '(e.g. exceeding max length or violating unique constraint) .'.format(repr(e))
            raise structs.StorageInternalError(message)

    def remove_session(self):
        # type: () -> None

        """Removes the current session.

        A session is stored in SQLAlchemy's ThreadLocalRegistry for each thread. This method
        closes and removes the session which is associated to the current thread. Particularly,
        under multi-thread use cases, it is important to call this method *from each thread*.
        Otherwise, all sessions and their associated DB connections are destructed by a thread
        that occasionally invoked the garbage collector. By default, it is not allowed to touch
        a SQLite connection from threads other than the thread that created the connection.
        Therefore, we need to explicitly close the connection from each thread.

        """

        self.scoped_session.remove()

    def __del__(self):
        # type: () -> None

        # This destructor calls remove_session to explicitly close the DB connection. We need this
        # because DB connections created in SQLAlchemy are not automatically closed by reference
        # counters, so it is not guaranteed that they are released by correct threads (for more
        # information, please see the docstring of remove_session).

        if hasattr(self, 'scoped_session'):
            self.remove_session()
