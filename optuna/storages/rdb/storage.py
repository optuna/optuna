import alembic.command
import alembic.config
import alembic.migration
import alembic.script
from collections import defaultdict
import copy
from datetime import datetime
import json
import logging
import os
import six
from sqlalchemy.engine import create_engine
from sqlalchemy.engine import Engine  # NOQA
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import orm
import sys
import threading
import uuid

import optuna
from optuna import distributions
from optuna.storages.base import BaseStorage
from optuna.storages.base import DEFAULT_STUDY_NAME_PREFIX
from optuna.storages.rdb import models
from optuna import structs
from optuna import types
from optuna import version

if types.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA


class RDBStorage(BaseStorage):
    """Storage class for RDB backend.

    This class is not supposed to be directly accessed by library users.

    Args:
        url: URL of the storage.
        engine_kwargs:
            A dictionary of keyword arguments that is passed to
            :func:`sqlalchemy.engine.create_engine`.
        enable_cache:
            Flag to control whether to enable storage layer caching.
            If this flag is set to :obj:`True` (the default), the finished trials are
            cached on memory and never re-fetched from the storage.
            Otherwise, the trials are fetched from the storage whenever they are needed.

    """

    def __init__(self, url, engine_kwargs=None, enable_cache=True, skip_compatibility_check=False):
        # type: (str, Optional[Dict[str, Any]], bool, bool) -> None

        engine_kwargs = engine_kwargs or {}

        url = self._fill_storage_url_template(url)

        try:
            self.engine = create_engine(url, **engine_kwargs)
        except ImportError as e:
            raise ImportError('Failed to import DB access module for the specified storage URL. '
                              'Please install appropriate one. (The actual import error is: ' +
                              str(e) + '.)')

        self.scoped_session = orm.scoped_session(orm.sessionmaker(bind=self.engine))
        models.BaseModel.metadata.create_all(self.engine)

        self.logger = optuna.logging.get_logger(__name__)

        self._version_manager = _VersionManager(url, self.engine, self.scoped_session)
        if not skip_compatibility_check:
            self._version_manager.check_table_schema_compatibility()

        self._finished_trials_cache = _FinishedTrialsCache(enable_cache)

    def create_new_study_id(self, study_name=None):
        # type: (Optional[str]) -> int

        session = self.scoped_session()

        if study_name is None:
            study_name = self._create_unique_study_name(session)

        study = models.StudyModel(study_name=study_name, direction=structs.StudyDirection.NOT_SET)
        session.add(study)
        if not self._commit_with_integrity_check(session):
            raise structs.DuplicatedStudyError(
                "Another study with name '{}' already exists. "
                "Please specify a different name, or reuse the existing one "
                "by setting `load_if_exists` (for Python API) or "
                "`--skip-if-exists` flag (for CLI).".format(study_name))

        self.logger.info('A new study created with name: {}'.format(study.study_name))

        return study.study_id

    @staticmethod
    def _create_unique_study_name(session):
        # type: (orm.Session) -> str

        while True:
            study_uuid = str(uuid.uuid4())
            study_name = DEFAULT_STUDY_NAME_PREFIX + study_uuid
            study = models.StudyModel.find_by_name(study_name, session)
            if study is None:
                break

        return study_name

    # TODO(sano): Prevent simultaneously setting different direction in distributed environments.
    def set_study_direction(self, study_id, direction):
        # type: (int, structs.StudyDirection) -> None

        session = self.scoped_session()

        study = models.StudyModel.find_or_raise_by_id(study_id, session)

        if study.direction != structs.StudyDirection.NOT_SET and study.direction != direction:
            raise ValueError('Cannot overwrite study direction from {} to {}.'.format(
                study.direction, direction))

        study.direction = direction

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

        self._commit_with_integrity_check(session)

    def set_study_system_attr(self, study_id, key, value):
        # type: (int, str, Any) -> None

        session = self.scoped_session()

        study = models.StudyModel.find_or_raise_by_id(study_id, session)
        attribute = models.StudySystemAttributeModel.find_by_study_and_key(study, key, session)
        if attribute is None:
            attribute = models.StudySystemAttributeModel(
                study_id=study_id, key=key, value_json=json.dumps(value))
            session.add(attribute)
        else:
            attribute.value_json = json.dumps(value)

        self._commit_with_integrity_check(session)

    def get_study_id_from_name(self, study_name):
        # type: (str) -> int

        session = self.scoped_session()

        study = models.StudyModel.find_or_raise_by_name(study_name, session)

        return study.study_id

    def get_study_id_from_trial_id(self, trial_id):
        # type: (int) -> int

        session = self.scoped_session()

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)

        return trial.study_id

    def get_study_name_from_id(self, study_id):
        # type: (int) -> str

        session = self.scoped_session()

        study = models.StudyModel.find_or_raise_by_id(study_id, session)

        return study.study_name

    def get_study_direction(self, study_id):
        # type: (int) -> structs.StudyDirection

        session = self.scoped_session()

        study = models.StudyModel.find_or_raise_by_id(study_id, session)

        return study.direction

    def get_study_user_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        session = self.scoped_session()

        attributes = models.StudyUserAttributeModel.where_study_id(study_id, session)

        return {attr.key: json.loads(attr.value_json) for attr in attributes}

    def get_study_system_attrs(self, study_id):
        # type: (int) -> Dict[str, Any]

        session = self.scoped_session()

        attributes = models.StudySystemAttributeModel.where_study_id(study_id, session)

        return {attr.key: json.loads(attr.value_json) for attr in attributes}

    def get_trial_user_attrs(self, trial_id):
        # type: (int) -> Dict[str, Any]

        session = self.scoped_session()

        attributes = models.TrialUserAttributeModel.where_trial_id(trial_id, session)

        return {attr.key: json.loads(attr.value_json) for attr in attributes}

    def get_trial_system_attrs(self, trial_id):
        # type: (int) -> Dict[str, Any]

        session = self.scoped_session()

        attributes = models.TrialSystemAttributeModel.where_trial_id(trial_id, session)

        return {attr.key: json.loads(attr.value_json) for attr in attributes}

    # TODO(sano): Optimize this method to reduce the number of queries.
    def get_all_study_summaries(self):
        # type: () -> List[structs.StudySummary]

        session = self.scoped_session()

        study_models = models.StudyModel.all(session)
        trial_models = models.TrialModel.all(session)
        param_models = models.TrialParamModel.all(session)
        value_models = models.TrialValueModel.all(session)
        trial_user_attribute_models = models.TrialUserAttributeModel.all(session)
        trial_system_attribute_models = models.TrialSystemAttributeModel.all(session)

        study_sumarries = []
        for study_model in study_models:
            # Filter model objects by study.
            study_trial_models = [t for t in trial_models if t.study_id == study_model.study_id]

            # Get best trial.
            completed_trial_models = [
                t for t in study_trial_models if t.state is structs.TrialState.COMPLETE
            ]
            best_trial = None
            if len(completed_trial_models) > 0:
                if study_model.direction == structs.StudyDirection.MAXIMIZE:
                    best_trial_model = max(completed_trial_models, key=lambda t: t.value)
                else:
                    best_trial_model = min(completed_trial_models, key=lambda t: t.value)

                best_param_models = [
                    p for p in param_models if p.trial_id == best_trial_model.trial_id
                ]
                best_value_models = [
                    v for v in value_models if v.trial_id == best_trial_model.trial_id
                ]
                best_trial_user_models = [
                    u for u in trial_user_attribute_models
                    if u.trial_id == best_trial_model.trial_id
                ]
                best_trial_system_models = [
                    s for s in trial_system_attribute_models
                    if s.trial_id == best_trial_model.trial_id
                ]

                # Merge model objects related to the best trial.
                best_trial = self._merge_trials_orm([best_trial_model], best_param_models,
                                                    best_value_models, best_trial_user_models,
                                                    best_trial_system_models)[0]

            # Find datetime_start.
            datetime_start = None
            if len(study_trial_models) > 0:
                datetime_start = min([t.datetime_start for t in study_trial_models])

            attributes = models.StudySystemAttributeModel.where_study_id(
                study_model.study_id, session)
            system_attrs = {attr.key: json.loads(attr.value_json) for attr in attributes}

            # Consolidate StudySummary.
            study_sumarries.append(
                structs.StudySummary(
                    study_id=study_model.study_id,
                    study_name=study_model.study_name,
                    direction=self.get_study_direction(study_model.study_id),
                    best_trial=best_trial,
                    user_attrs=self.get_study_user_attrs(study_model.study_id),
                    system_attrs=system_attrs,
                    n_trials=len(study_trial_models),
                    datetime_start=datetime_start))

        return study_sumarries

    def create_new_trial_id(self, study_id):
        # type: (int) -> int

        session = self.scoped_session()

        trial = models.TrialModel(study_id=study_id, state=structs.TrialState.RUNNING)

        session.add(trial)
        self._commit(session)

        self._create_new_trial_number(trial.trial_id)

        return trial.trial_id

    def _create_new_trial_number(self, trial_id):
        # type: (int) -> int

        session = self.scoped_session()
        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)

        trial_number = trial.count_past_trials(session)
        self.set_trial_system_attr(trial.trial_id, '_number', trial_number)

        return trial_number

    def set_trial_state(self, trial_id, state):
        # type: (int, structs.TrialState) -> None

        session = self.scoped_session()

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        self.check_trial_is_updatable(trial_id, trial.state)

        trial.state = state
        if state.is_finished():
            trial.datetime_complete = datetime.now()

        self._commit(session)

    def set_trial_param(self, trial_id, param_name, param_value_internal, distribution):
        # type: (int, str, float, distributions.BaseDistribution) -> bool

        session = self.scoped_session()

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        self.check_trial_is_updatable(trial_id, trial.state)

        trial_param = \
            models.TrialParamModel.find_by_trial_and_param_name(trial, param_name, session)

        if trial_param is not None:
            # Raise error in case distribution is incompatible.
            distributions.check_distribution_compatibility(
                distributions.json_to_distribution(trial_param.distribution_json), distribution)

            # Return False when distribution is compatible but parameter has already been set.
            return False

        param = models.TrialParamModel(
            trial_id=trial_id,
            param_name=param_name,
            param_value=param_value_internal,
            distribution_json=distributions.distribution_to_json(distribution))

        param.check_and_add(session)
        commit_success = self._commit_with_integrity_check(session)

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
        self.check_trial_is_updatable(trial_id, trial.state)

        trial.value = value

        self._commit(session)

    def set_trial_intermediate_value(self, trial_id, step, intermediate_value):
        # type: (int, int, float) -> bool

        session = self.scoped_session()

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        self.check_trial_is_updatable(trial_id, trial.state)

        trial_value = models.TrialValueModel.find_by_trial_and_step(trial, step, session)
        if trial_value is not None:
            return False

        trial_value = models.TrialValueModel(
            trial_id=trial_id, step=step, value=intermediate_value)

        session.add(trial_value)
        commit_success = self._commit_with_integrity_check(session)

        return commit_success

    def set_trial_user_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        session = self.scoped_session()

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        self.check_trial_is_updatable(trial_id, trial.state)

        attribute = models.TrialUserAttributeModel.find_by_trial_and_key(trial, key, session)
        if attribute is None:
            attribute = models.TrialUserAttributeModel(
                trial_id=trial_id, key=key, value_json=json.dumps(value))
            session.add(attribute)
        else:
            attribute.value_json = json.dumps(value)

        self._commit_with_integrity_check(session)

    def set_trial_system_attr(self, trial_id, key, value):
        # type: (int, str, Any) -> None

        session = self.scoped_session()

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        if key == '_number':
            # `_number` attribute may be set even after a trial is finished.
            # This happens if the trial was created before v0.9.0,
            # where a trial didn't have `_number` attribute.
            # In this case, `check_trial_is_updatable` is skipped to avoid the `RuntimeError`.
            #
            # TODO(ohta): Remove this workaround when `number` field is added to `TrialModel`.
            pass
        else:
            self.check_trial_is_updatable(trial_id, trial.state)

        attribute = models.TrialSystemAttributeModel.find_by_trial_and_key(trial, key, session)
        if attribute is None:
            attribute = models.TrialSystemAttributeModel(
                trial_id=trial_id, key=key, value_json=json.dumps(value))
            session.add(attribute)
        else:
            attribute.value_json = json.dumps(value)

        self._commit_with_integrity_check(session)

    def get_trial_number_from_id(self, trial_id):
        # type: (int) -> int

        trial_number = self.get_trial_system_attrs(trial_id).get('_number')
        if trial_number is None:
            # If a study is created by optuna<=0.8.0, trial number is not found.
            # Create new one.
            return self._create_new_trial_number(trial_id)
        return trial_number

    def get_trial(self, trial_id):
        # type: (int) -> structs.FrozenTrial

        cached_trial = self._finished_trials_cache.get_cached_trial(trial_id)
        if cached_trial is not None:
            return copy.deepcopy(cached_trial)

        session = self.scoped_session()

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        params = models.TrialParamModel.where_trial(trial, session)
        values = models.TrialValueModel.where_trial(trial, session)
        user_attributes = models.TrialUserAttributeModel.where_trial(trial, session)
        system_attributes = models.TrialSystemAttributeModel.where_trial(trial, session)

        frozen_trial = self._merge_trials_orm([trial], params, values, user_attributes,
                                              system_attributes)[0]

        self._finished_trials_cache.cache_trial_if_finished(frozen_trial)

        return frozen_trial

    def get_all_trials(self, study_id):
        # type: (int) -> List[structs.FrozenTrial]

        if self._finished_trials_cache.is_empty():
            trials = self._get_all_trials_without_cache(study_id)
            for trial in trials:
                self._finished_trials_cache.cache_trial_if_finished(trial)

            return trials

        trial_ids = self._get_all_trial_ids(study_id)
        trials = [self.get_trial(trial_id) for trial_id in trial_ids]
        return trials

    def _get_all_trial_ids(self, study_id):
        # type: (int) -> List[int]

        session = self.scoped_session()
        study = models.StudyModel.find_or_raise_by_id(study_id, session)
        return models.TrialModel.get_all_trial_ids_where_study(study, session)

    def _get_all_trials_without_cache(self, study_id):
        # type: (int) -> List[structs.FrozenTrial]

        session = self.scoped_session()

        study = models.StudyModel.find_or_raise_by_id(study_id, session)
        trials = models.TrialModel.where_study(study, session)
        params = models.TrialParamModel.where_study(study, session)
        values = models.TrialValueModel.where_study(study, session)
        user_attributes = models.TrialUserAttributeModel.where_study(study, session)
        system_attributes = models.TrialSystemAttributeModel.where_study(study, session)

        return self._merge_trials_orm(trials, params, values, user_attributes, system_attributes)

    def get_n_trials(self, study_id, state=None):
        # type: (int, Optional[structs.TrialState]) -> int

        session = self.scoped_session()
        study = models.StudyModel.find_or_raise_by_id(study_id, session)
        return models.TrialModel.count(session, study, state)

    def _merge_trials_orm(
            self,
            trials,  # type: List[models.TrialModel]
            trial_params,  # type: List[models.TrialParamModel]
            trial_intermediate_values,  # type: List[models.TrialValueModel]
            trial_user_attrs,  # type: List[models.TrialUserAttributeModel]
            trial_system_attrs  # type: List[models.TrialSystemAttributeModel]
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

        id_to_user_attrs = \
            defaultdict(list)  # type: Dict[int, List[models.TrialUserAttributeModel]]
        for user_attr in trial_user_attrs:
            id_to_user_attrs[user_attr.trial_id].append(user_attr)

        id_to_system_attrs = \
            defaultdict(list)  # type: Dict[int, List[models.TrialSystemAttributeModel]]
        for system_attr in trial_system_attrs:
            id_to_system_attrs[system_attr.trial_id].append(system_attr)

        temp_trials = []
        for trial_id, trial in id_to_trial.items():
            params = {}
            params_in_internal_repr = {}
            param_distributions = {}
            for param in id_to_params[trial_id]:
                distribution = distributions.json_to_distribution(param.distribution_json)
                params[param.param_name] = distribution.to_external_repr(param.param_value)
                params_in_internal_repr[param.param_name] = param.param_value
                param_distributions[param.param_name] = distribution

            intermediate_values = {}
            for value in id_to_values[trial_id]:
                intermediate_values[value.step] = value.value

            user_attrs = {}
            for user_attr in id_to_user_attrs[trial_id]:
                user_attrs[user_attr.key] = json.loads(user_attr.value_json)

            system_attrs = {}
            for system_attr in id_to_system_attrs[trial_id]:
                system_attrs[system_attr.key] = json.loads(system_attr.value_json)

            # `-1` is a dummy value.
            # It will be replaced by a proper value before returned to the caller.
            #
            # TODO(ohta): Use trial.number after TrialModel.number is added.
            trial_number = -1

            temp_trials.append(
                structs.FrozenTrial(
                    number=trial_number,
                    state=trial.state,
                    params=params,
                    distributions=param_distributions,
                    user_attrs=user_attrs,
                    system_attrs=system_attrs,
                    value=trial.value,
                    intermediate_values=intermediate_values,
                    params_in_internal_repr=params_in_internal_repr,
                    datetime_start=trial.datetime_start,
                    datetime_complete=trial.datetime_complete,
                    trial_id=trial_id))

        result = []
        for temp_trial in temp_trials:
            # [NOTE]
            # We set actual trial numbers here to avoid calling `self.get_trial_number_from_id()`
            # within the above loop.
            #
            # This is because `self.get_trial_number_from_id()` may call `session.commit()`
            # internally, which causes unintended changes of the states of `trials`.
            # (see https://github.com/pfnet/optuna/pull/349#issuecomment-475086642 for details)
            trial_number = self.get_trial_number_from_id(temp_trial.trial_id)
            result.append(temp_trial._replace(number=trial_number))

        return result

    @staticmethod
    def _fill_storage_url_template(template):
        # type: (str) -> str

        return template.format(SCHEMA_VERSION=models.SCHEMA_VERSION)

    @staticmethod
    def _commit_with_integrity_check(session):
        # type: (orm.Session) -> bool

        try:
            session.commit()
        except IntegrityError as e:
            logger = optuna.logging.get_logger(__name__)
            logger.debug(
                'Ignoring {}. This happens due to a timing issue among threads/processes/nodes. '
                'Another one might have committed a record with the same key(s).'.format(repr(e)))
            session.rollback()
            return False

        return True

    @staticmethod
    def _commit(session):
        # type: (orm.Session) -> None

        try:
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            message = \
                'An exception is raised during the commit. ' \
                'This typically happens due to invalid data in the commit, ' \
                'e.g. exceeding max length. ' \
                '(The actual exception is as follows: {})'.format(repr(e))
            six.reraise(structs.StorageInternalError, structs.StorageInternalError(message),
                        sys.exc_info()[2])

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

    def upgrade(self):
        # type: () -> None
        """Upgrade the storage schema."""

        self._version_manager.upgrade()

    def get_current_version(self):
        # type: () -> str
        """Return the schema version currently used by this storage."""

        return self._version_manager.get_current_version()

    def get_head_version(self):
        # type: () -> str
        """Return the latest schema version."""

        return self._version_manager.get_head_version()

    def get_all_versions(self):
        # type: () -> List[str]
        """Return the schema version list."""

        return self._version_manager.get_all_versions()


class _VersionManager(object):
    def __init__(self, url, engine, scoped_session):
        # type: (str, Engine, orm.scoped_session) -> None

        self.url = url
        self.engine = engine
        self.scoped_session = scoped_session

        self._init_version_info_model()
        self._init_alembic()

    def _init_version_info_model(self):
        # type: () -> None

        session = self.scoped_session()

        version_info = models.VersionInfoModel.find(session)
        if version_info is not None:
            return

        version_info = models.VersionInfoModel(
            schema_version=models.SCHEMA_VERSION, library_version=version.__version__)

        session.add(version_info)
        RDBStorage._commit_with_integrity_check(session)

    def _init_alembic(self):
        # type: () -> None

        logging.getLogger('alembic').setLevel(logging.WARN)

        context = alembic.migration.MigrationContext.configure(self.engine.connect())
        is_initialized = context.get_current_revision() is not None

        if is_initialized:
            # The `alembic_version` table already exists and is not empty.
            return

        if self._is_alembic_supported():
            revision = self.get_head_version()
        else:
            # The storage has been created before alembic is introduced.
            revision = self._get_base_version()

        self._set_alembic_revision(revision)

    def _set_alembic_revision(self, revision):
        # type: (str) -> None

        context = alembic.migration.MigrationContext.configure(self.engine.connect())
        script = self._create_alembic_script()
        context.stamp(script, revision)

    def check_table_schema_compatibility(self):
        # type: () -> None

        session = self.scoped_session()

        # NOTE: After invocation of `_init_version_info_model` method,
        #       it is ensured that a `VersionInfoModel` entry exists.
        version_info = models.VersionInfoModel.find(session)
        assert version_info is not None

        current_version = self.get_current_version()
        head_version = self.get_head_version()
        if current_version == head_version:
            return

        message = 'The runtime optuna version {} is no longer compatible with the table schema ' \
                  '(set up by optuna {}). '.format(version.__version__,
                                                   version_info.library_version)
        known_versions = self.get_all_versions()
        if current_version in known_versions:
            message += 'Please execute `$ optuna storage upgrade --storage $STORAGE_URL` ' \
                       'for upgrading the storage.'
        else:
            message += 'Please try updating optuna to the latest version by '\
                       '`$ pip install -U optuna`.'

        raise RuntimeError(message)

    def get_current_version(self):
        # type: () -> str

        context = alembic.migration.MigrationContext.configure(self.engine.connect())
        version = context.get_current_revision()
        assert version is not None

        return version

    def get_head_version(self):
        # type: () -> str

        script = self._create_alembic_script()
        return script.get_current_head()

    def _get_base_version(self):
        # type: () -> str

        script = self._create_alembic_script()
        return script.get_base()

    def get_all_versions(self):
        # type: () -> List[str]

        script = self._create_alembic_script()
        return [r.revision for r in script.walk_revisions()]

    def upgrade(self):
        # type: () -> None

        config = self._create_alembic_config()
        alembic.command.upgrade(config, 'head')

    def _is_alembic_supported(self):
        # type: () -> bool

        session = self.scoped_session()

        version_info = models.VersionInfoModel.find(session)
        if version_info is None:
            # `None` means this storage was created just now.
            return True

        return version_info.schema_version == models.SCHEMA_VERSION

    def _create_alembic_script(self):
        # type: () -> alembic.script.ScriptDirectory

        config = self._create_alembic_config()
        script = alembic.script.ScriptDirectory.from_config(config)
        return script

    def _create_alembic_config(self):
        # type: () -> alembic.config.Config

        alembic_dir = os.path.join(os.path.dirname(__file__), 'alembic')

        config = alembic.config.Config(os.path.join(os.path.dirname(__file__), 'alembic.ini'))
        config.set_main_option('script_location', escape_alembic_config_value(alembic_dir))
        config.set_main_option('sqlalchemy.url', escape_alembic_config_value(self.url))
        return config


class _FinishedTrialsCache(object):
    def __init__(self, enabled):
        # type: (bool) -> None

        self._finished_trials = {}  # type: Dict[int, structs.FrozenTrial]
        self._enabled = enabled
        self._lock = threading.Lock()

    def is_empty(self):
        # type: () -> bool

        if not self._enabled:
            return True

        with self._lock:
            return len(self._finished_trials) == 0

    def cache_trial_if_finished(self, trial):
        # type: (structs.FrozenTrial) -> None

        if not self._enabled:
            return

        if trial.state.is_finished():
            with self._lock:
                self._finished_trials[trial.trial_id] = copy.deepcopy(trial)

    def get_cached_trial(self, trial_id):
        # type: (int) -> Optional[structs.FrozenTrial]

        if not self._enabled:
            return None

        with self._lock:
            return self._finished_trials.get(trial_id)


def escape_alembic_config_value(value):
    # type: (str) -> str

    # We must escape '%' in a value string because the character
    # is regarded as the trigger of variable expansion.
    # Please see the documentation of `configparser.BasicInterpolation` for more details.
    return value.replace('%', '%%')
