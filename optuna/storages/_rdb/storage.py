from contextlib import contextmanager
import copy
from datetime import datetime
import json
import logging
import os
from typing import Any
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
import uuid

import alembic.command
import alembic.config
import alembic.migration
import alembic.script
from sqlalchemy import orm
from sqlalchemy.engine import create_engine
from sqlalchemy.engine import Engine  # NOQA
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import OperationalError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import functions

import optuna
from optuna import distributions
from optuna import version
from optuna._study_direction import StudyDirection
from optuna._study_summary import StudySummary
from optuna.storages._base import BaseStorage
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.storages._rdb import models
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = optuna.logging.get_logger(__name__)


@contextmanager
def _create_scoped_session(
    scoped_session: orm.scoped_session,
    ignore_integrity_error: bool = False,
) -> Generator[orm.Session, None, None]:
    session = scoped_session()
    try:
        yield session
        session.commit()
    except IntegrityError as e:
        session.rollback()
        if ignore_integrity_error:
            _logger.debug(
                "Ignoring {}. This happens due to a timing issue among threads/processes/nodes. "
                "Another one might have committed a record with the same key(s).".format(repr(e))
            )
        else:
            raise
    except SQLAlchemyError as e:
        session.rollback()
        message = (
            "An exception is raised during the commit. "
            "This typically happens due to invalid data in the commit, "
            "e.g. exceeding max length. "
        )
        raise optuna.exceptions.StorageInternalError(message) from e
    except Exception:
        session.rollback()
        raise


class RDBStorage(BaseStorage):
    """Storage class for RDB backend.

    Note that library users can instantiate this class, but the attributes
    provided by this class are not supposed to be directly accessed by them.

    Example:

        Create an :class:`~optuna.storages.RDBStorage` instance with customized
        ``pool_size`` and ``timeout`` settings.

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_uniform("x", -100, 100)
                return x ** 2


            storage = optuna.storages.RDBStorage(
                url="sqlite:///:memory:",
                engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
            )

            study = optuna.create_study(storage=storage)
            study.optimize(objective, n_trials=10)

    Args:
        url: URL of the storage.
        engine_kwargs:
            A dictionary of keyword arguments that is passed to
            `sqlalchemy.engine.create_engine`_ function.
        skip_compatibility_check:
            Flag to skip schema compatibility check if set to True.

    .. _sqlalchemy.engine.create_engine:
        https://docs.sqlalchemy.org/en/latest/core/engines.html#sqlalchemy.create_engine

    .. note::
        If you use MySQL, `pool_pre_ping`_ will be set to :obj:`True` by default to prevent
        connection timeout. You can turn it off with ``engine_kwargs['pool_pre_ping']=False``, but
        it is recommended to keep the setting if execution time of your objective function is
        longer than the `wait_timeout` of your MySQL configuration.

    .. _pool_pre_ping:
        https://docs.sqlalchemy.org/en/13/core/engines.html#sqlalchemy.create_engine.params.
        pool_pre_ping
    """

    def __init__(
        self,
        url: str,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        skip_compatibility_check: bool = False,
    ) -> None:

        self.engine_kwargs = engine_kwargs or {}
        self.url = self._fill_storage_url_template(url)
        self.skip_compatibility_check = skip_compatibility_check

        self._set_default_engine_kwargs_for_mysql(url, self.engine_kwargs)

        try:
            self.engine = create_engine(self.url, **self.engine_kwargs)
        except ImportError as e:
            raise ImportError(
                "Failed to import DB access module for the specified storage URL. "
                "Please install appropriate one."
            ) from e

        self.scoped_session = orm.scoped_session(orm.sessionmaker(bind=self.engine))
        models.BaseModel.metadata.create_all(self.engine)

        self._version_manager = _VersionManager(self.url, self.engine, self.scoped_session)
        if not skip_compatibility_check:
            self._version_manager.check_table_schema_compatibility()

    def __getstate__(self) -> Dict[Any, Any]:

        state = self.__dict__.copy()
        del state["scoped_session"]
        del state["engine"]
        del state["_version_manager"]
        return state

    def __setstate__(self, state: Dict[Any, Any]) -> None:

        self.__dict__.update(state)
        try:
            self.engine = create_engine(self.url, **self.engine_kwargs)
        except ImportError as e:
            raise ImportError(
                "Failed to import DB access module for the specified storage URL. "
                "Please install appropriate one."
            ) from e

        self.scoped_session = orm.scoped_session(orm.sessionmaker(bind=self.engine))
        models.BaseModel.metadata.create_all(self.engine)
        self._version_manager = _VersionManager(self.url, self.engine, self.scoped_session)
        if not self.skip_compatibility_check:
            self._version_manager.check_table_schema_compatibility()

    def create_new_study(self, study_name: Optional[str] = None) -> int:

        try:
            with _create_scoped_session(self.scoped_session) as session:
                if study_name is None:
                    study_name = self._create_unique_study_name(session)

                direction = models.StudyDirectionModel(
                    direction=StudyDirection.NOT_SET, objective=0
                )
                study = models.StudyModel(study_name=study_name, directions=[direction])
                session.add(study)
        except IntegrityError:
            raise optuna.exceptions.DuplicatedStudyError(
                "Another study with name '{}' already exists. "
                "Please specify a different name, or reuse the existing one "
                "by setting `load_if_exists` (for Python API) or "
                "`--skip-if-exists` flag (for CLI).".format(study_name)
            )

        _logger.info("A new study created in RDB with name: {}".format(study.study_name))

        return study.study_id

    def delete_study(self, study_id: int) -> None:

        with _create_scoped_session(self.scoped_session, True) as session:
            study = models.StudyModel.find_or_raise_by_id(study_id, session)
            session.delete(study)

    @staticmethod
    def _create_unique_study_name(session: orm.Session) -> str:

        while True:
            study_uuid = str(uuid.uuid4())
            study_name = DEFAULT_STUDY_NAME_PREFIX + study_uuid
            study = models.StudyModel.find_by_name(study_name, session)
            if study is None:
                break

        return study_name

    # TODO(sano): Prevent simultaneously setting different direction in distributed environments.
    def set_study_directions(self, study_id: int, directions: Sequence[StudyDirection]) -> None:

        with _create_scoped_session(self.scoped_session) as session:
            study = models.StudyModel.find_or_raise_by_id(study_id, session)
            directions = list(directions)
            current_directions = [
                d.direction for d in models.StudyDirectionModel.where_study_id(study_id, session)
            ]
            if (
                len(current_directions) > 0
                and current_directions[0] != StudyDirection.NOT_SET
                and current_directions != directions
            ):
                raise ValueError(
                    "Cannot overwrite study direction from {} to {}.".format(
                        current_directions, directions
                    )
                )

            for objective, d in enumerate(directions):
                direction_model = models.StudyDirectionModel.find_by_study_and_objective(
                    study, objective, session
                )
                if direction_model is None:
                    direction_model = models.StudyDirectionModel(
                        study_id=study_id, objective=objective, direction=d
                    )
                    session.add(direction_model)
                else:
                    direction_model.direction = d

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:

        with _create_scoped_session(self.scoped_session, True) as session:
            study = models.StudyModel.find_or_raise_by_id(study_id, session)
            attribute = models.StudyUserAttributeModel.find_by_study_and_key(study, key, session)
            if attribute is None:
                attribute = models.StudyUserAttributeModel(
                    study_id=study_id, key=key, value_json=json.dumps(value)
                )
                session.add(attribute)
            else:
                attribute.value_json = json.dumps(value)

    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:

        with _create_scoped_session(self.scoped_session, True) as session:
            study = models.StudyModel.find_or_raise_by_id(study_id, session)
            attribute = models.StudySystemAttributeModel.find_by_study_and_key(study, key, session)
            if attribute is None:
                attribute = models.StudySystemAttributeModel(
                    study_id=study_id, key=key, value_json=json.dumps(value)
                )
                session.add(attribute)
            else:
                attribute.value_json = json.dumps(value)

    def get_study_id_from_name(self, study_name: str) -> int:

        with _create_scoped_session(self.scoped_session) as session:
            study = models.StudyModel.find_or_raise_by_name(study_name, session)

        return study.study_id

    def get_study_id_from_trial_id(self, trial_id: int) -> int:

        with _create_scoped_session(self.scoped_session) as session:
            trial = models.TrialModel.find_or_raise_by_id(trial_id, session)

        return trial.study_id

    def get_study_name_from_id(self, study_id: int) -> str:

        with _create_scoped_session(self.scoped_session) as session:
            study = models.StudyModel.find_or_raise_by_id(study_id, session)

        return study.study_name

    def get_study_directions(self, study_id: int) -> List[StudyDirection]:

        with _create_scoped_session(self.scoped_session) as session:
            study = models.StudyModel.find_or_raise_by_id(study_id, session)
            directions = [d.direction for d in study.directions]

        return directions

    def get_study_user_attrs(self, study_id: int) -> Dict[str, Any]:

        with _create_scoped_session(self.scoped_session) as session:
            # Ensure that that study exists.
            models.StudyModel.find_or_raise_by_id(study_id, session)
            attributes = models.StudyUserAttributeModel.where_study_id(study_id, session)
            user_attrs = {attr.key: json.loads(attr.value_json) for attr in attributes}

        return user_attrs

    def get_study_system_attrs(self, study_id: int) -> Dict[str, Any]:

        with _create_scoped_session(self.scoped_session) as session:
            # Ensure that that study exists.
            models.StudyModel.find_or_raise_by_id(study_id, session)
            attributes = models.StudySystemAttributeModel.where_study_id(study_id, session)
            system_attrs = {attr.key: json.loads(attr.value_json) for attr in attributes}

        return system_attrs

    def get_trial_user_attrs(self, trial_id: int) -> Dict[str, Any]:

        with _create_scoped_session(self.scoped_session) as session:
            # Ensure trial exists.
            models.TrialModel.find_or_raise_by_id(trial_id, session)

            attributes = models.TrialUserAttributeModel.where_trial_id(trial_id, session)
            user_attrs = {attr.key: json.loads(attr.value_json) for attr in attributes}

        return user_attrs

    def get_trial_system_attrs(self, trial_id: int) -> Dict[str, Any]:

        with _create_scoped_session(self.scoped_session) as session:
            # Ensure trial exists.
            models.TrialModel.find_or_raise_by_id(trial_id, session)

            attributes = models.TrialSystemAttributeModel.where_trial_id(trial_id, session)
            system_attrs = {attr.key: json.loads(attr.value_json) for attr in attributes}

        return system_attrs

    def get_all_study_summaries(self) -> List[StudySummary]:

        with _create_scoped_session(self.scoped_session) as session:
            summarized_trial = (
                session.query(
                    models.TrialModel.study_id,
                    functions.min(models.TrialModel.datetime_start).label("datetime_start"),
                    functions.count(models.TrialModel.trial_id).label("n_trial"),
                )
                .group_by(models.TrialModel.study_id)
                .with_labels()
                .subquery()
            )
            study_summary_stmt = session.query(
                models.StudyModel.study_id,
                models.StudyModel.study_name,
                summarized_trial.c.datetime_start,
                functions.coalesce(summarized_trial.c.n_trial, 0).label("n_trial"),
            ).select_from(orm.outerjoin(models.StudyModel, summarized_trial))

            study_summary = study_summary_stmt.all()
            study_summaries = []
            for study in study_summary:
                directions = [
                    d.direction
                    for d in models.StudyDirectionModel.where_study_id(study.study_id, session)
                ]
                best_trial: Optional[models.TrialModel] = None
                try:
                    if len(directions) > 1:
                        raise ValueError
                    elif directions[0] == StudyDirection.MAXIMIZE:
                        best_trial = models.TrialModel.find_max_value_trial(
                            study.study_id, 0, session
                        )
                    else:
                        best_trial = models.TrialModel.find_min_value_trial(
                            study.study_id, 0, session
                        )
                except ValueError:
                    best_trial_frozen: Optional[FrozenTrial] = None
                if best_trial:
                    value = models.TrialValueModel.find_by_trial_and_objective(
                        best_trial, 0, session
                    )
                    assert value
                    params = (
                        session.query(
                            models.TrialParamModel.param_name,
                            models.TrialParamModel.param_value,
                            models.TrialParamModel.distribution_json,
                        )
                        .filter(models.TrialParamModel.trial_id == best_trial.trial_id)
                        .all()
                    )
                    param_dict = {}
                    param_distributions = {}
                    for param in params:
                        distribution = distributions.json_to_distribution(param.distribution_json)
                        param_dict[param.param_name] = distribution.to_external_repr(
                            param.param_value
                        )
                        param_distributions[param.param_name] = distribution
                    user_attrs = models.TrialUserAttributeModel.where_trial_id(
                        best_trial.trial_id, session
                    )
                    system_attrs = models.TrialSystemAttributeModel.where_trial_id(
                        best_trial.trial_id, session
                    )
                    intermediate = models.TrialIntermediateValueModel.where_trial_id(
                        best_trial.trial_id, session
                    )
                    best_trial_frozen = FrozenTrial(
                        best_trial.number,
                        TrialState.COMPLETE,
                        value.value,
                        best_trial.datetime_start,
                        best_trial.datetime_complete,
                        param_dict,
                        param_distributions,
                        {i.key: json.loads(i.value_json) for i in user_attrs},
                        {i.key: json.loads(i.value_json) for i in system_attrs},
                        {value.step: value.intermediate_value for value in intermediate},
                        best_trial.trial_id,
                    )
                user_attrs = models.StudyUserAttributeModel.where_study_id(study.study_id, session)
                system_attrs = models.StudySystemAttributeModel.where_study_id(
                    study.study_id, session
                )
                study_summaries.append(
                    StudySummary(
                        study_name=study.study_name,
                        direction=None,
                        directions=directions,
                        best_trial=best_trial_frozen,
                        user_attrs={i.key: json.loads(i.value_json) for i in user_attrs},
                        system_attrs={i.key: json.loads(i.value_json) for i in system_attrs},
                        n_trials=study.n_trial,
                        datetime_start=study.datetime_start,
                        study_id=study.study_id,
                    )
                )

        return study_summaries

    def create_new_trial(self, study_id: int, template_trial: Optional[FrozenTrial] = None) -> int:

        return self._create_new_trial(study_id, template_trial)._trial_id

    def _create_new_trial(
        self, study_id: int, template_trial: Optional[FrozenTrial] = None
    ) -> FrozenTrial:
        """Create a new trial and returns its trial_id and a :class:`~optuna.trial.FrozenTrial`.

        Args:
            study_id:
                Study id.
            template_trial:
                A :class:`~optuna.trial.FrozenTrial` with default values for trial attributes.

        Returns:
            A :class:`~optuna.trial.FrozenTrial` instance.

        """

        # Retry a couple of times. Deadlocks may occur in distributed environments.
        n_retries = 0
        while True:
            try:
                with _create_scoped_session(self.scoped_session) as session:
                    # Ensure that that study exists.
                    #
                    # Locking within a study is necessary since the creation of a trial is not an
                    # atomic operation. More precisely, the trial number computed in
                    # `_get_prepared_new_trial` is prone to race conditions without this lock.
                    models.StudyModel.find_or_raise_by_id(study_id, session, for_update=True)

                    trial = self._get_prepared_new_trial(study_id, template_trial, session)
                break  # Successfully created trial.
            except OperationalError:
                if n_retries > 2:
                    raise

            n_retries += 1

        if template_trial:
            frozen = copy.deepcopy(template_trial)
            frozen.number = trial.number
            frozen.datetime_start = trial.datetime_start
            frozen._trial_id = trial.trial_id
        else:
            frozen = FrozenTrial(
                number=trial.number,
                state=trial.state,
                value=None,
                values=None,
                datetime_start=trial.datetime_start,
                datetime_complete=None,
                params={},
                distributions={},
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=trial.trial_id,
            )

        return frozen

    def _get_prepared_new_trial(
        self, study_id: int, template_trial: Optional[FrozenTrial], session: orm.Session
    ) -> models.TrialModel:
        if template_trial is None:
            trial = models.TrialModel(study_id=study_id, number=None, state=TrialState.RUNNING)
        else:
            # Because only `RUNNING` trials can be updated,
            # we temporarily set the state of the new trial to `RUNNING`.
            # After all fields of the trial have been updated,
            # the state is set to `template_trial.state`.
            temp_state = TrialState.RUNNING

            trial = models.TrialModel(
                study_id=study_id,
                number=None,
                state=temp_state,
                datetime_start=template_trial.datetime_start,
                datetime_complete=template_trial.datetime_complete,
            )

        session.add(trial)

        # Flush the session cache to reflect the above addition operation to
        # the current RDB transaction.
        #
        # Without flushing, the following operations (e.g, `_set_trial_param_without_commit`)
        # will fail because the target trial doesn't exist in the storage yet.
        session.flush()

        if template_trial is not None:
            if template_trial.values is not None and len(template_trial.values) > 1:
                for objective, value in enumerate(template_trial.values):
                    self._set_trial_value_without_commit(session, trial.trial_id, objective, value)
            elif template_trial.value is not None:
                self._set_trial_value_without_commit(
                    session, trial.trial_id, 0, template_trial.value
                )

            for param_name, param_value in template_trial.params.items():
                distribution = template_trial.distributions[param_name]
                param_value_in_internal_repr = distribution.to_internal_repr(param_value)
                self._set_trial_param_without_commit(
                    session, trial.trial_id, param_name, param_value_in_internal_repr, distribution
                )

            for key, value in template_trial.user_attrs.items():
                self._set_trial_user_attr_without_commit(session, trial.trial_id, key, value)

            for key, value in template_trial.system_attrs.items():
                self._set_trial_system_attr_without_commit(session, trial.trial_id, key, value)

            for step, intermediate_value in template_trial.intermediate_values.items():
                self._set_trial_intermediate_value_without_commit(
                    session, trial.trial_id, step, intermediate_value
                )

            trial.state = template_trial.state

        trial.number = trial.count_past_trials(session)
        session.add(trial)

        return trial

    def _update_trial(
        self,
        trial_id: int,
        state: Optional[TrialState] = None,
        values: Optional[Sequence[float]] = None,
        intermediate_values: Optional[Dict[int, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        distributions_: Optional[Dict[str, distributions.BaseDistribution]] = None,
        user_attrs: Optional[Dict[str, Any]] = None,
        system_attrs: Optional[Dict[str, Any]] = None,
        datetime_complete: Optional[datetime] = None,
    ) -> bool:
        """Sync latest trial updates to a database.

        Args:
            trial_id:
                Trial id of the trial to update.
            state:
                New state. None when there are no changes.
            values:
                New value. None when there are no changes.
            intermediate_values:
                New intermediate values. None when there are no updates.
            params:
                New parameter dictionary. None when there are no updates.
            distributions_:
                New parameter distributions. None when there are no updates.
            user_attrs:
                New user_attr. None when there are no updates.
            system_attrs:
                New system_attr. None when there are no updates.
            datetime_complete:
                Completion time of the trial. Set if and only if this method
                change the state of trial into one of the finished states.

        Returns:
            True when success.

        """

        with _create_scoped_session(self.scoped_session) as session:
            trial_model = models.TrialModel.find_or_raise_by_id(trial_id, session)
            if trial_model.state.is_finished():
                raise RuntimeError("Cannot change attributes of finished trial.")
            if (
                state
                and trial_model.state != state
                and state == TrialState.RUNNING
                and trial_model.state != TrialState.WAITING
            ):
                return False

            if state:
                trial_model.state = state

            if datetime_complete:
                trial_model.datetime_complete = datetime_complete

            if values is not None:
                trial_values = models.TrialValueModel.where_trial_id(trial_id, session)
                if len(trial_values) > 0:
                    for objective in range(len(values)):
                        trial_values[objective].value = values[objective]
                        session.add(trial_values[objective])
                else:
                    for objective in range(len(values)):
                        trial_model.values.extend(
                            [models.TrialValueModel(objective=objective, value=values[objective])]
                        )

            if user_attrs:
                trial_user_attrs = models.TrialUserAttributeModel.where_trial_id(trial_id, session)
                trial_user_attrs_dict = {attr.key: attr for attr in trial_user_attrs}
                for k, v in user_attrs.items():
                    if k in trial_user_attrs_dict:
                        trial_user_attrs_dict[k].value_json = json.dumps(v)
                        session.add(trial_user_attrs_dict[k])
                trial_model.user_attributes.extend(
                    models.TrialUserAttributeModel(key=k, value_json=json.dumps(v))
                    for k, v in user_attrs.items()
                    if k not in trial_user_attrs_dict
                )

            if system_attrs:
                trial_system_attrs = models.TrialSystemAttributeModel.where_trial_id(
                    trial_id, session
                )
                trial_system_attrs_dict = {attr.key: attr for attr in trial_system_attrs}
                for k, v in system_attrs.items():
                    if k in trial_system_attrs_dict:
                        trial_system_attrs_dict[k].value_json = json.dumps(v)
                        session.add(trial_system_attrs_dict[k])
                trial_model.system_attributes.extend(
                    models.TrialSystemAttributeModel(key=k, value_json=json.dumps(v))
                    for k, v in system_attrs.items()
                    if k not in trial_system_attrs_dict
                )

            if intermediate_values:
                trial_intermediate_values = models.TrialIntermediateValueModel.where_trial_id(
                    trial_id, session
                )
                intermediate_values_dict = {v.step: v for v in trial_intermediate_values}
                for s, v in intermediate_values.items():
                    if s in intermediate_values_dict:
                        intermediate_values_dict[s].intermediate_value = v
                        session.add(intermediate_values_dict[s])
                trial_model.intermediate_values.extend(
                    models.TrialIntermediateValueModel(step=s, intermediate_value=v)
                    for s, v in intermediate_values.items()
                    if s not in intermediate_values_dict
                )

            if params and distributions_:
                trial_param = models.TrialParamModel.where_trial_id(trial_id, session)
                trial_param_dict = {attr.param_name: attr for attr in trial_param}
                for name, v in params.items():
                    if name in trial_param_dict:
                        trial_param_dict[
                            name
                        ].distribution_json = distributions.distribution_to_json(
                            distributions_[name]
                        )
                        trial_param_dict[name].param_value = v
                        session.add(trial_param_dict[name])
                trial_model.params.extend(
                    models.TrialParamModel(
                        param_name=param_name,
                        param_value=param_value,
                        distribution_json=distributions.distribution_to_json(
                            distributions_[param_name]
                        ),
                    )
                    for param_name, param_value in params.items()
                    if param_name not in trial_param_dict
                )

            session.add(trial_model)

        return True

    def set_trial_state(self, trial_id: int, state: TrialState) -> bool:

        try:
            with _create_scoped_session(self.scoped_session) as session:
                trial = models.TrialModel.find_or_raise_by_id(trial_id, session, for_update=True)
                self.check_trial_is_updatable(trial_id, trial.state)

                if state == TrialState.RUNNING and trial.state != TrialState.WAITING:
                    return False

                trial.state = state
                if state.is_finished():
                    trial.datetime_complete = datetime.now()
        except IntegrityError:
            return False
        return True

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: distributions.BaseDistribution,
    ) -> None:

        with _create_scoped_session(self.scoped_session, True) as session:
            self._set_trial_param_without_commit(
                session, trial_id, param_name, param_value_internal, distribution
            )

    def _set_trial_param_without_commit(
        self,
        session: orm.Session,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: distributions.BaseDistribution,
    ) -> None:

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        self.check_trial_is_updatable(trial_id, trial.state)

        trial_param = models.TrialParamModel.find_by_trial_and_param_name(
            trial, param_name, session
        )

        if trial_param is not None:
            # Raise error in case distribution is incompatible.
            distributions.check_distribution_compatibility(
                distributions.json_to_distribution(trial_param.distribution_json), distribution
            )

            trial_param.param_value = param_value_internal
            trial_param.distribution_json = distributions.distribution_to_json(distribution)
        else:
            trial_param = models.TrialParamModel(
                trial_id=trial_id,
                param_name=param_name,
                param_value=param_value_internal,
                distribution_json=distributions.distribution_to_json(distribution),
            )

            trial_param.check_and_add(session)

    def _check_and_set_param_distribution(
        self,
        study_id: int,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: distributions.BaseDistribution,
    ) -> None:

        with _create_scoped_session(self.scoped_session) as session:
            # Acquire lock.
            #
            # Assume that study exists.
            models.StudyModel.find_or_raise_by_id(study_id, session, for_update=True)

            models.TrialParamModel(
                trial_id=trial_id,
                param_name=param_name,
                param_value=param_value_internal,
                distribution_json=distributions.distribution_to_json(distribution),
            ).check_and_add(session)

    def get_trial_param(self, trial_id: int, param_name: str) -> float:

        with _create_scoped_session(self.scoped_session) as session:
            trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
            trial_param = models.TrialParamModel.find_or_raise_by_trial_and_param_name(
                trial, param_name, session
            )

        return trial_param.param_value

    def set_trial_values(self, trial_id: int, values: Sequence[float]) -> None:

        with _create_scoped_session(self.scoped_session) as session:
            trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
            self.check_trial_is_updatable(trial_id, trial.state)
            for objective, v in enumerate(values):
                self._set_trial_value_without_commit(session, trial_id, objective, v)

    def _set_trial_value_without_commit(
        self, session: orm.Session, trial_id: int, objective: int, value: float
    ) -> None:

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        self.check_trial_is_updatable(trial_id, trial.state)

        trial_value = models.TrialValueModel.find_by_trial_and_objective(trial, objective, session)
        if trial_value is None:
            trial_value = models.TrialValueModel(
                trial_id=trial_id, objective=objective, value=value
            )
            session.add(trial_value)
        else:
            trial_value.value = value

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:

        with _create_scoped_session(self.scoped_session, True) as session:
            self._set_trial_intermediate_value_without_commit(
                session, trial_id, step, intermediate_value
            )

    def _set_trial_intermediate_value_without_commit(
        self, session: orm.Session, trial_id: int, step: int, intermediate_value: float
    ) -> None:

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        self.check_trial_is_updatable(trial_id, trial.state)

        trial_intermediate_value = models.TrialIntermediateValueModel.find_by_trial_and_step(
            trial, step, session
        )
        if trial_intermediate_value is None:
            trial_intermediate_value = models.TrialIntermediateValueModel(
                trial_id=trial_id, step=step, intermediate_value=intermediate_value
            )
            session.add(trial_intermediate_value)
        else:
            trial_intermediate_value.intermediate_value = intermediate_value

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:

        with _create_scoped_session(self.scoped_session, True) as session:
            self._set_trial_user_attr_without_commit(session, trial_id, key, value)

    def _set_trial_user_attr_without_commit(
        self, session: orm.Session, trial_id: int, key: str, value: Any
    ) -> None:

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        self.check_trial_is_updatable(trial_id, trial.state)

        attribute = models.TrialUserAttributeModel.find_by_trial_and_key(trial, key, session)
        if attribute is None:
            attribute = models.TrialUserAttributeModel(
                trial_id=trial_id, key=key, value_json=json.dumps(value)
            )
            session.add(attribute)
        else:
            attribute.value_json = json.dumps(value)

    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:

        with _create_scoped_session(self.scoped_session, True) as session:
            self._set_trial_system_attr_without_commit(session, trial_id, key, value)

    def _set_trial_system_attr_without_commit(
        self, session: orm.Session, trial_id: int, key: str, value: Any
    ) -> None:

        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        self.check_trial_is_updatable(trial_id, trial.state)

        attribute = models.TrialSystemAttributeModel.find_by_trial_and_key(trial, key, session)
        if attribute is None:
            attribute = models.TrialSystemAttributeModel(
                trial_id=trial_id, key=key, value_json=json.dumps(value)
            )
            session.add(attribute)
        else:
            attribute.value_json = json.dumps(value)

    def get_trial_id_from_study_id_trial_number(self, study_id: int, trial_number: int) -> int:

        with _create_scoped_session(self.scoped_session) as session:
            trial_id = (
                session.query(models.TrialModel.trial_id)
                .filter(
                    models.TrialModel.number == trial_number,
                    models.TrialModel.study_id == study_id,
                )
                .one_or_none()
            )
            if trial_id is None:
                raise KeyError(
                    "No trial with trial number {} exists in study with study_id {}.".format(
                        trial_number, study_id
                    )
                )
            return trial_id[0]

    def get_trial_number_from_id(self, trial_id: int) -> int:

        trial_number = self.get_trial(trial_id).number
        return trial_number

    def get_trial(self, trial_id: int) -> FrozenTrial:

        with _create_scoped_session(self.scoped_session) as session:
            trial_model = models.TrialModel.find_or_raise_by_id(trial_id, session)
            frozen_trial = self._build_frozen_trial_from_trial_model(trial_model)

        return frozen_trial

    def get_all_trials(
        self,
        study_id: int,
        deepcopy: bool = True,
        states: Optional[Tuple[TrialState, ...]] = None,
    ) -> List[FrozenTrial]:

        trials = self._get_trials(study_id, states, set())

        return copy.deepcopy(trials) if deepcopy else trials

    def _get_trials(
        self,
        study_id: int,
        states: Optional[Tuple[TrialState, ...]],
        excluded_trial_ids: Set[int],
    ) -> List[FrozenTrial]:

        with _create_scoped_session(self.scoped_session) as session:
            # Ensure that the study exists.
            models.StudyModel.find_or_raise_by_id(study_id, session)
            query = session.query(models.TrialModel.trial_id).filter(
                models.TrialModel.study_id == study_id
            )

            if states is not None:
                query = query.filter(models.TrialModel.state.in_(states))

            trial_ids = query.all()

            trial_ids = set(
                trial_id_tuple[0]
                for trial_id_tuple in trial_ids
                if trial_id_tuple[0] not in excluded_trial_ids
            )
            try:
                trial_models = (
                    session.query(models.TrialModel)
                    .options(orm.selectinload(models.TrialModel.params))
                    .options(orm.selectinload(models.TrialModel.values))
                    .options(orm.selectinload(models.TrialModel.user_attributes))
                    .options(orm.selectinload(models.TrialModel.system_attributes))
                    .options(orm.selectinload(models.TrialModel.intermediate_values))
                    .filter(
                        models.TrialModel.trial_id.in_(trial_ids),
                        models.TrialModel.study_id == study_id,
                    )
                    .all()
                )
            except OperationalError as e:
                # Likely exceeding the number of maximum allowed variables using IN.
                # This number differ between database dialects. For SQLite for instance, see
                # https://www.sqlite.org/limits.html and the section describing
                # SQLITE_MAX_VARIABLE_NUMBER.

                _logger.warning(
                    "Caught an error from sqlalchemy: {}. Falling back to a slower alternative. "
                    "".format(str(e))
                )

                trial_models = (
                    session.query(models.TrialModel)
                    .options(orm.selectinload(models.TrialModel.params))
                    .options(orm.selectinload(models.TrialModel.values))
                    .options(orm.selectinload(models.TrialModel.user_attributes))
                    .options(orm.selectinload(models.TrialModel.system_attributes))
                    .options(orm.selectinload(models.TrialModel.intermediate_values))
                    .filter(models.TrialModel.study_id == study_id)
                    .all()
                )
                trial_models = [t for t in trial_models if t.trial_id in trial_ids]

            trials = [self._build_frozen_trial_from_trial_model(trial) for trial in trial_models]

        return trials

    @staticmethod
    def _build_frozen_trial_from_trial_model(trial: models.TrialModel) -> FrozenTrial:

        values: Optional[List[float]]
        if trial.values:
            values = [0 for _ in trial.values]
            for value_model in trial.values:
                values[value_model.objective] = value_model.value
        else:
            values = None

        return FrozenTrial(
            number=trial.number,
            state=trial.state,
            value=None,
            values=values,
            datetime_start=trial.datetime_start,
            datetime_complete=trial.datetime_complete,
            params={
                p.param_name: distributions.json_to_distribution(
                    p.distribution_json
                ).to_external_repr(p.param_value)
                for p in trial.params
            },
            distributions={
                p.param_name: distributions.json_to_distribution(p.distribution_json)
                for p in trial.params
            },
            user_attrs={attr.key: json.loads(attr.value_json) for attr in trial.user_attributes},
            system_attrs={
                attr.key: json.loads(attr.value_json) for attr in trial.system_attributes
            },
            intermediate_values={v.step: v.intermediate_value for v in trial.intermediate_values},
            trial_id=trial.trial_id,
        )

    def get_best_trial(self, study_id: int) -> FrozenTrial:

        with _create_scoped_session(self.scoped_session) as session:
            _directions = self.get_study_directions(study_id)
            if len(_directions) > 1:
                raise ValueError(
                    "Best trial can be obtained only for single-objective optimization."
                )
            direction = _directions[0]

            if direction == StudyDirection.MAXIMIZE:
                trial = models.TrialModel.find_max_value_trial(study_id, 0, session)
            else:
                trial = models.TrialModel.find_min_value_trial(study_id, 0, session)

        return self.get_trial(trial.trial_id)

    def read_trials_from_remote_storage(self, study_id: int) -> None:
        # Make sure that the given study exists.
        with _create_scoped_session(self.scoped_session) as session:
            models.StudyModel.find_or_raise_by_id(study_id, session)

    @staticmethod
    def _set_default_engine_kwargs_for_mysql(url: str, engine_kwargs: Dict[str, Any]) -> None:

        # Skip if RDB is not MySQL.
        if not url.startswith("mysql"):
            return

        # Do not overwrite value.
        if "pool_pre_ping" in engine_kwargs:
            return

        # If True, the connection pool checks liveness of connections at every checkout.
        # Without this option, trials that take longer than `wait_timeout` may cause connection
        # errors. For further details, please refer to the following document:
        # https://docs.sqlalchemy.org/en/13/core/pooling.html#pool-disconnects-pessimistic
        engine_kwargs["pool_pre_ping"] = True
        _logger.debug("pool_pre_ping=True was set to engine_kwargs to prevent connection timeout.")

    @staticmethod
    def _fill_storage_url_template(template: str) -> str:

        return template.format(SCHEMA_VERSION=models.SCHEMA_VERSION)

    def remove_session(self) -> None:
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

    def upgrade(self) -> None:
        """Upgrade the storage schema."""

        self._version_manager.upgrade()

    def get_current_version(self) -> str:
        """Return the schema version currently used by this storage."""

        return self._version_manager.get_current_version()

    def get_head_version(self) -> str:
        """Return the latest schema version."""

        return self._version_manager.get_head_version()

    def get_all_versions(self) -> List[str]:
        """Return the schema version list."""

        return self._version_manager.get_all_versions()


class _VersionManager(object):
    def __init__(self, url: str, engine: Engine, scoped_session: orm.scoped_session) -> None:

        self.url = url
        self.engine = engine
        self.scoped_session = scoped_session
        self._init_version_info_model()
        self._init_alembic()

    def _init_version_info_model(self) -> None:

        with _create_scoped_session(self.scoped_session, True) as session:
            version_info = models.VersionInfoModel.find(session)
            if version_info is not None:
                return

            version_info = models.VersionInfoModel(
                schema_version=models.SCHEMA_VERSION, library_version=version.__version__
            )
            session.add(version_info)

    def _init_alembic(self) -> None:

        logging.getLogger("alembic").setLevel(logging.WARN)

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

    def _set_alembic_revision(self, revision: str) -> None:

        context = alembic.migration.MigrationContext.configure(self.engine.connect())
        script = self._create_alembic_script()
        context.stamp(script, revision)

    def check_table_schema_compatibility(self) -> None:

        with _create_scoped_session(self.scoped_session) as session:
            # NOTE: After invocation of `_init_version_info_model` method,
            #       it is ensured that a `VersionInfoModel` entry exists.
            version_info = models.VersionInfoModel.find(session)

        assert version_info is not None

        current_version = self.get_current_version()
        head_version = self.get_head_version()
        if current_version == head_version:
            return

        message = (
            "The runtime optuna version {} is no longer compatible with the table schema "
            "(set up by optuna {}). ".format(version.__version__, version_info.library_version)
        )
        known_versions = self.get_all_versions()
        if current_version in known_versions:
            message += (
                "Please execute `$ optuna storage upgrade --storage $STORAGE_URL` "
                "for upgrading the storage."
            )
        else:
            message += (
                "Please try updating optuna to the latest version by `$ pip install -U optuna`."
            )

        raise RuntimeError(message)

    def get_current_version(self) -> str:

        context = alembic.migration.MigrationContext.configure(self.engine.connect())
        version = context.get_current_revision()
        assert version is not None

        return version

    def get_head_version(self) -> str:

        script = self._create_alembic_script()
        return script.get_current_head()

    def _get_base_version(self) -> str:

        script = self._create_alembic_script()
        return script.get_base()

    def get_all_versions(self) -> List[str]:

        script = self._create_alembic_script()
        return [r.revision for r in script.walk_revisions()]

    def upgrade(self) -> None:

        config = self._create_alembic_config()
        alembic.command.upgrade(config, "head")

    def _is_alembic_supported(self) -> bool:

        with _create_scoped_session(self.scoped_session) as session:
            version_info = models.VersionInfoModel.find(session)

        if version_info is None:
            # `None` means this storage was created just now.
            return True

        return version_info.schema_version == models.SCHEMA_VERSION

    def _create_alembic_script(self) -> alembic.script.ScriptDirectory:

        config = self._create_alembic_config()
        script = alembic.script.ScriptDirectory.from_config(config)
        return script

    def _create_alembic_config(self) -> alembic.config.Config:

        alembic_dir = os.path.join(os.path.dirname(__file__), "alembic")

        config = alembic.config.Config(os.path.join(os.path.dirname(__file__), "alembic.ini"))
        config.set_main_option("script_location", escape_alembic_config_value(alembic_dir))
        config.set_main_option("sqlalchemy.url", escape_alembic_config_value(self.url))
        return config


def escape_alembic_config_value(value: str) -> str:

    # We must escape '%' in a value string because the character
    # is regarded as the trigger of variable expansion.
    # Please see the documentation of `configparser.BasicInterpolation` for more details.
    return value.replace("%", "%%")
