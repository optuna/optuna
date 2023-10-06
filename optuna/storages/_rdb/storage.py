from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import copy
from datetime import datetime
import json
import logging
import os
from typing import Any
from typing import Callable
from typing import Container
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TYPE_CHECKING
import uuid

import optuna
from optuna import distributions
from optuna import version
from optuna._imports import _LazyImport
from optuna._typing import JSONSerializable
from optuna.storages._base import BaseStorage
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.storages._heartbeat import BaseHeartbeat
from optuna.storages._rdb.models import TrialValueModel
from optuna.study._frozen import FrozenStudy
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    import alembic.command as alembic_command
    import alembic.config as alembic_config
    import alembic.migration as alembic_migration
    import alembic.script as alembic_script
    import sqlalchemy
    import sqlalchemy.exc as sqlalchemy_exc
    import sqlalchemy.orm as sqlalchemy_orm
    import sqlalchemy.sql.functions as sqlalchemy_sql_functions

    from optuna.storages._rdb import models
else:
    alembic_command = _LazyImport("alembic.command")
    alembic_config = _LazyImport("alembic.config")
    alembic_migration = _LazyImport("alembic.migration")
    alembic_script = _LazyImport("alembic.script")

    sqlalchemy = _LazyImport("sqlalchemy")
    sqlalchemy_exc = _LazyImport("sqlalchemy.exc")
    sqlalchemy_orm = _LazyImport("sqlalchemy.orm")
    sqlalchemy_sql_functions = _LazyImport("sqlalchemy.sql.functions")

    models = _LazyImport("optuna.storages._rdb.models")


_logger = optuna.logging.get_logger(__name__)


@contextmanager
def _create_scoped_session(
    scoped_session: "sqlalchemy_orm.scoped_session",
    ignore_integrity_error: bool = False,
) -> Generator["sqlalchemy_orm.Session", None, None]:
    session = scoped_session()
    try:
        yield session
        session.commit()
    except sqlalchemy_exc.IntegrityError as e:
        session.rollback()
        if ignore_integrity_error:
            _logger.debug(
                "Ignoring {}. This happens due to a timing issue among threads/processes/nodes. "
                "Another one might have committed a record with the same key(s).".format(repr(e))
            )
        else:
            raise
    except sqlalchemy_exc.SQLAlchemyError as e:
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
    finally:
        session.close()


class RDBStorage(BaseStorage, BaseHeartbeat):
    """Storage class for RDB backend.

    Note that library users can instantiate this class, but the attributes
    provided by this class are not supposed to be directly accessed by them.

    Example:

        Create an :class:`~optuna.storages.RDBStorage` instance with customized
        ``pool_size`` and ``timeout`` settings.

        .. testcode::

            import optuna


            def objective(trial):
                x = trial.suggest_float("x", -100, 100)
                return x**2


            storage = optuna.storages.RDBStorage(
                url="sqlite:///:memory:",
                engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
            )

            study = optuna.create_study(storage=storage)
            study.optimize(objective, n_trials=10)

    Args:
        url:
            URL of the storage.
        engine_kwargs:
            A dictionary of keyword arguments that is passed to
            `sqlalchemy.engine.create_engine`_ function.
        skip_compatibility_check:
            Flag to skip schema compatibility check if set to :obj:`True`.
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

        skip_table_creation:
            Flag to skip table creation if set to :obj:`True`.

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

    .. note::
        We would never recommend SQLite3 for parallel optimization.
        Please see the FAQ :ref:`sqlite_concurrency` for details.

    .. note::
        Mainly in a cluster environment, running trials are often killed unexpectedly.
        If you want to detect a failure of trials, please use the heartbeat
        mechanism. Set ``heartbeat_interval``, ``grace_period``, and ``failed_trial_callback``
        appropriately according to your use case. For more details, please refer to the
        :ref:`tutorial <heartbeat_monitoring>` and `Example page
        <https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_checkpoint.py>`_.

    .. seealso::
        You can use :class:`~optuna.storages.RetryFailedTrialCallback` to automatically retry
        failed trials detected by heartbeat.

    """

    def __init__(
        self,
        url: str,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        skip_compatibility_check: bool = False,
        *,
        heartbeat_interval: Optional[int] = None,
        grace_period: Optional[int] = None,
        failed_trial_callback: Optional[
            Callable[["optuna.study.Study", FrozenTrial], None]
        ] = None,
        skip_table_creation: bool = False,
    ) -> None:
        self.engine_kwargs = engine_kwargs or {}
        self.url = self._fill_storage_url_template(url)
        self.skip_compatibility_check = skip_compatibility_check
        if heartbeat_interval is not None and heartbeat_interval <= 0:
            raise ValueError("The value of `heartbeat_interval` should be a positive integer.")
        if grace_period is not None and grace_period <= 0:
            raise ValueError("The value of `grace_period` should be a positive integer.")
        self.heartbeat_interval = heartbeat_interval
        self.grace_period = grace_period
        self.failed_trial_callback = failed_trial_callback

        self._set_default_engine_kwargs_for_mysql(url, self.engine_kwargs)

        try:
            self.engine = sqlalchemy.engine.create_engine(self.url, **self.engine_kwargs)
        except ImportError as e:
            raise ImportError(
                "Failed to import DB access module for the specified storage URL. "
                "Please install appropriate one."
            ) from e

        self.scoped_session = sqlalchemy_orm.scoped_session(
            sqlalchemy_orm.sessionmaker(bind=self.engine)
        )
        if not skip_table_creation:
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
            self.engine = sqlalchemy.engine.create_engine(self.url, **self.engine_kwargs)
        except ImportError as e:
            raise ImportError(
                "Failed to import DB access module for the specified storage URL. "
                "Please install appropriate one."
            ) from e

        self.scoped_session = sqlalchemy_orm.scoped_session(
            sqlalchemy_orm.sessionmaker(bind=self.engine)
        )
        models.BaseModel.metadata.create_all(self.engine)
        self._version_manager = _VersionManager(self.url, self.engine, self.scoped_session)
        if not self.skip_compatibility_check:
            self._version_manager.check_table_schema_compatibility()

    def create_new_study(
        self, directions: Sequence[StudyDirection], study_name: Optional[str] = None
    ) -> int:
        try:
            with _create_scoped_session(self.scoped_session) as session:
                if study_name is None:
                    study_name = self._create_unique_study_name(session)

                direction_models = [
                    models.StudyDirectionModel(objective=objective, direction=d)
                    for objective, d in enumerate(list(directions))
                ]

                session.add(models.StudyModel(study_name=study_name, directions=direction_models))

        except sqlalchemy_exc.IntegrityError:
            raise optuna.exceptions.DuplicatedStudyError(
                "Another study with name '{}' already exists. "
                "Please specify a different name, or reuse the existing one "
                "by setting `load_if_exists` (for Python API) or "
                "`--skip-if-exists` flag (for CLI).".format(study_name)
            )

        _logger.info("A new study created in RDB with name: {}".format(study_name))

        return self.get_study_id_from_name(study_name)

    def delete_study(self, study_id: int) -> None:
        with _create_scoped_session(self.scoped_session, True) as session:
            study = models.StudyModel.find_or_raise_by_id(study_id, session)
            session.delete(study)

    @staticmethod
    def _create_unique_study_name(session: "sqlalchemy_orm.Session") -> str:
        while True:
            study_uuid = str(uuid.uuid4())
            study_name = DEFAULT_STUDY_NAME_PREFIX + study_uuid
            study = models.StudyModel.find_by_name(study_name, session)
            if study is None:
                break

        return study_name

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

    def set_study_system_attr(self, study_id: int, key: str, value: JSONSerializable) -> None:
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
            study_id = study.study_id

        return study_id

    def get_study_name_from_id(self, study_id: int) -> str:
        with _create_scoped_session(self.scoped_session) as session:
            study = models.StudyModel.find_or_raise_by_id(study_id, session)
            study_name = study.study_name

        return study_name

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

    def get_all_studies(self) -> List[FrozenStudy]:
        with _create_scoped_session(self.scoped_session) as session:
            studies = (
                session.query(
                    models.StudyModel.study_id,
                    models.StudyModel.study_name,
                )
                .order_by(models.StudyModel.study_id)
                .all()
            )

            _directions = defaultdict(list)
            for direction_model in session.query(models.StudyDirectionModel).all():
                _directions[direction_model.study_id].append(direction_model.direction)

            _user_attrs = defaultdict(list)
            for attribute_model in session.query(models.StudyUserAttributeModel).all():
                _user_attrs[attribute_model.study_id].append(attribute_model)

            _system_attrs = defaultdict(list)
            for attribute_model in session.query(models.StudySystemAttributeModel).all():
                _system_attrs[attribute_model.study_id].append(attribute_model)

            frozen_studies = []
            for study in studies:
                directions = _directions[study.study_id]
                user_attrs = _user_attrs.get(study.study_id, [])
                system_attrs = _system_attrs.get(study.study_id, [])
                frozen_studies.append(
                    FrozenStudy(
                        study_name=study.study_name,
                        direction=None,
                        directions=directions,
                        user_attrs={i.key: json.loads(i.value_json) for i in user_attrs},
                        system_attrs={i.key: json.loads(i.value_json) for i in system_attrs},
                        study_id=study.study_id,
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

        # Retry a couple of times. Deadlocks may occur in distributed environments.
        n_retries = 0
        with _create_scoped_session(self.scoped_session) as session:
            while True:
                try:
                    # Ensure that that study exists.
                    #
                    # Locking within a study is necessary since the creation of a trial is not an
                    # atomic operation. More precisely, the trial number computed in
                    # `_get_prepared_new_trial` is prone to race conditions without this lock.
                    models.StudyModel.find_or_raise_by_id(study_id, session, for_update=True)

                    trial = self._get_prepared_new_trial(study_id, template_trial, session)
                    break  # Successfully created trial.
                except sqlalchemy_exc.OperationalError:
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
        self,
        study_id: int,
        template_trial: Optional[FrozenTrial],
        session: "sqlalchemy_orm.Session",
    ) -> "models.TrialModel":
        if template_trial is None:
            trial = models.TrialModel(
                study_id=study_id,
                number=None,
                state=TrialState.RUNNING,
                datetime_start=datetime.now(),
            )
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
        session: "sqlalchemy_orm.Session",
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
            param_value = trial_param.param_value

        return param_value

    def set_trial_state_values(
        self, trial_id: int, state: TrialState, values: Optional[Sequence[float]] = None
    ) -> bool:
        try:
            with _create_scoped_session(self.scoped_session) as session:
                trial = models.TrialModel.find_or_raise_by_id(trial_id, session, for_update=True)
                self.check_trial_is_updatable(trial_id, trial.state)

                if values is not None:
                    for objective, v in enumerate(values):
                        self._set_trial_value_without_commit(session, trial_id, objective, v)

                if state == TrialState.RUNNING and trial.state != TrialState.WAITING:
                    return False

                trial.state = state

                if state == TrialState.RUNNING:
                    trial.datetime_start = datetime.now()

                if state.is_finished():
                    trial.datetime_complete = datetime.now()
        except sqlalchemy_exc.IntegrityError:
            return False
        return True

    def _set_trial_value_without_commit(
        self, session: "sqlalchemy_orm.Session", trial_id: int, objective: int, value: float
    ) -> None:
        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        self.check_trial_is_updatable(trial_id, trial.state)
        stored_value, value_type = TrialValueModel.value_to_stored_repr(value)

        trial_value = models.TrialValueModel.find_by_trial_and_objective(trial, objective, session)
        if trial_value is None:
            trial_value = models.TrialValueModel(
                trial_id=trial_id, objective=objective, value=stored_value, value_type=value_type
            )
            session.add(trial_value)
        else:
            trial_value.value = stored_value
            trial_value.value_type = value_type

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:
        with _create_scoped_session(self.scoped_session, True) as session:
            self._set_trial_intermediate_value_without_commit(
                session, trial_id, step, intermediate_value
            )

    def _set_trial_intermediate_value_without_commit(
        self,
        session: "sqlalchemy_orm.Session",
        trial_id: int,
        step: int,
        intermediate_value: float,
    ) -> None:
        trial = models.TrialModel.find_or_raise_by_id(trial_id, session)
        self.check_trial_is_updatable(trial_id, trial.state)

        (
            stored_value,
            value_type,
        ) = models.TrialIntermediateValueModel.intermediate_value_to_stored_repr(
            intermediate_value
        )
        trial_intermediate_value = models.TrialIntermediateValueModel.find_by_trial_and_step(
            trial, step, session
        )
        if trial_intermediate_value is None:
            trial_intermediate_value = models.TrialIntermediateValueModel(
                trial_id=trial_id,
                step=step,
                intermediate_value=stored_value,
                intermediate_value_type=value_type,
            )
            session.add(trial_intermediate_value)
        else:
            trial_intermediate_value.intermediate_value = stored_value
            trial_intermediate_value.intermediate_value_type = value_type

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:
        with _create_scoped_session(self.scoped_session, True) as session:
            self._set_trial_user_attr_without_commit(session, trial_id, key, value)

    def _set_trial_user_attr_without_commit(
        self, session: "sqlalchemy_orm.Session", trial_id: int, key: str, value: Any
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

    def set_trial_system_attr(self, trial_id: int, key: str, value: JSONSerializable) -> None:
        with _create_scoped_session(self.scoped_session, True) as session:
            self._set_trial_system_attr_without_commit(session, trial_id, key, value)

    def _set_trial_system_attr_without_commit(
        self, session: "sqlalchemy_orm.Session", trial_id: int, key: str, value: JSONSerializable
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

    def get_trial(self, trial_id: int) -> FrozenTrial:
        with _create_scoped_session(self.scoped_session) as session:
            trial_model = models.TrialModel.find_or_raise_by_id(trial_id, session)
            frozen_trial = self._build_frozen_trial_from_trial_model(trial_model)

        return frozen_trial

    def get_all_trials(
        self,
        study_id: int,
        deepcopy: bool = True,
        states: Optional[Container[TrialState]] = None,
    ) -> List[FrozenTrial]:
        trials = self._get_trials(study_id, states, set())

        return copy.deepcopy(trials) if deepcopy else trials

    def _get_trials(
        self,
        study_id: int,
        states: Optional[Container[TrialState]],
        excluded_trial_ids: Set[int],
    ) -> List[FrozenTrial]:
        with _create_scoped_session(self.scoped_session) as session:
            # Ensure that the study exists.
            models.StudyModel.find_or_raise_by_id(study_id, session)
            query = session.query(models.TrialModel.trial_id).filter(
                models.TrialModel.study_id == study_id
            )

            if states is not None:
                # This assertion is for type checkers, since `states` is required to be Container
                # in the base class while `models.TrialModel.state.in_` requires Iterable.
                assert isinstance(states, Iterable)
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
                    .options(sqlalchemy_orm.selectinload(models.TrialModel.params))
                    .options(sqlalchemy_orm.selectinload(models.TrialModel.values))
                    .options(sqlalchemy_orm.selectinload(models.TrialModel.user_attributes))
                    .options(sqlalchemy_orm.selectinload(models.TrialModel.system_attributes))
                    .options(sqlalchemy_orm.selectinload(models.TrialModel.intermediate_values))
                    .filter(
                        models.TrialModel.trial_id.in_(trial_ids),
                        models.TrialModel.study_id == study_id,
                    )
                    .order_by(models.TrialModel.trial_id)
                    .all()
                )
            except sqlalchemy_exc.OperationalError as e:
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
                    .options(sqlalchemy_orm.selectinload(models.TrialModel.params))
                    .options(sqlalchemy_orm.selectinload(models.TrialModel.values))
                    .options(sqlalchemy_orm.selectinload(models.TrialModel.user_attributes))
                    .options(sqlalchemy_orm.selectinload(models.TrialModel.system_attributes))
                    .options(sqlalchemy_orm.selectinload(models.TrialModel.intermediate_values))
                    .filter(models.TrialModel.study_id == study_id)
                    .order_by(models.TrialModel.trial_id)
                    .all()
                )
                trial_models = [t for t in trial_models if t.trial_id in trial_ids]

            trials = [self._build_frozen_trial_from_trial_model(trial) for trial in trial_models]

        return trials

    def _build_frozen_trial_from_trial_model(self, trial: "models.TrialModel") -> FrozenTrial:
        values: Optional[List[float]]
        if trial.values:
            values = [0 for _ in trial.values]
            for value_model in trial.values:
                values[value_model.objective] = TrialValueModel.stored_repr_to_value(
                    value_model.value, value_model.value_type
                )
        else:
            values = None

        params = sorted(trial.params, key=lambda p: p.param_id)

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
                for p in params
            },
            distributions={
                p.param_name: distributions.json_to_distribution(p.distribution_json)
                for p in params
            },
            user_attrs={attr.key: json.loads(attr.value_json) for attr in trial.user_attributes},
            system_attrs={
                attr.key: json.loads(attr.value_json) for attr in trial.system_attributes
            },
            intermediate_values={
                v.step: models.TrialIntermediateValueModel.stored_repr_to_intermediate_value(
                    v.intermediate_value, v.intermediate_value_type
                )
                for v in trial.intermediate_values
            },
            trial_id=trial.trial_id,
        )

    def get_best_trial(self, study_id: int) -> FrozenTrial:
        with _create_scoped_session(self.scoped_session) as session:
            _directions = self.get_study_directions(study_id)
            if len(_directions) > 1:
                raise RuntimeError(
                    "Best trial can be obtained only for single-objective optimization."
                )
            direction = _directions[0]

            if direction == StudyDirection.MAXIMIZE:
                trial = models.TrialModel.find_max_value_trial(study_id, 0, session)
            else:
                trial = models.TrialModel.find_min_value_trial(study_id, 0, session)
            trial_id = trial.trial_id

        return self.get_trial(trial_id)

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

    def record_heartbeat(self, trial_id: int) -> None:
        with _create_scoped_session(self.scoped_session, True) as session:
            heartbeat = models.TrialHeartbeatModel.where_trial_id(trial_id, session)
            if heartbeat is None:
                heartbeat = models.TrialHeartbeatModel(trial_id=trial_id)
                session.add(heartbeat)
            else:
                heartbeat.heartbeat = session.execute(sqlalchemy.func.now()).scalar()

    def _get_stale_trial_ids(self, study_id: int) -> List[int]:
        assert self.heartbeat_interval is not None
        if self.grace_period is None:
            grace_period = 2 * self.heartbeat_interval
        else:
            grace_period = self.grace_period
        stale_trial_ids = []

        with _create_scoped_session(self.scoped_session, True) as session:
            current_heartbeat = session.execute(sqlalchemy.func.now()).scalar()
            assert current_heartbeat is not None
            # Added the following line to prevent mixing of timezone-aware and timezone-naive
            # `datetime` in PostgreSQL. See
            # https://github.com/optuna/optuna/pull/2190#issuecomment-766605088 for details
            current_heartbeat = current_heartbeat.replace(tzinfo=None)

            running_trials = (
                session.query(models.TrialModel)
                .options(sqlalchemy_orm.selectinload(models.TrialModel.heartbeats))
                .filter(models.TrialModel.state == TrialState.RUNNING)
                .filter(models.TrialModel.study_id == study_id)
                .all()
            )
            for trial in running_trials:
                if len(trial.heartbeats) == 0:
                    continue
                assert len(trial.heartbeats) == 1
                heartbeat = trial.heartbeats[0].heartbeat
                if (current_heartbeat - heartbeat).seconds > grace_period:
                    stale_trial_ids.append(trial.trial_id)

        return stale_trial_ids

    def get_heartbeat_interval(self) -> Optional[int]:
        return self.heartbeat_interval

    def get_failed_trial_callback(
        self,
    ) -> Optional[Callable[["optuna.study.Study", FrozenTrial], None]]:
        return self.failed_trial_callback


class _VersionManager:
    def __init__(
        self,
        url: str,
        engine: "sqlalchemy.engine.Engine",
        scoped_session: "sqlalchemy_orm.scoped_session",
    ) -> None:
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
                schema_version=models.SCHEMA_VERSION,
                library_version=version.__version__,
            )
            session.add(version_info)

    def _init_alembic(self) -> None:
        logging.getLogger("alembic").setLevel(logging.WARN)

        with self.engine.connect() as connection:
            context = alembic_migration.MigrationContext.configure(connection)
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
        with self.engine.connect() as connection:
            context = alembic_migration.MigrationContext.configure(connection)
            with connection.begin():
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
        with self.engine.connect() as connection:
            context = alembic_migration.MigrationContext.configure(connection)
            version = context.get_current_revision()
        assert version is not None

        return version

    def get_head_version(self) -> str:
        script = self._create_alembic_script()
        current_head = script.get_current_head()
        assert current_head is not None
        return current_head

    def _get_base_version(self) -> str:
        script = self._create_alembic_script()
        base = script.get_base()
        assert base is not None, "There should be exactly one base, i.e. v0.9.0.a."
        return base

    def get_all_versions(self) -> List[str]:
        script = self._create_alembic_script()
        return [r.revision for r in script.walk_revisions()]

    def upgrade(self) -> None:
        config = self._create_alembic_config()
        alembic_command.upgrade(config, "head")

        with _create_scoped_session(self.scoped_session, True) as session:
            version_info = models.VersionInfoModel.find(session)
            assert version_info is not None
            version_info.schema_version = models.SCHEMA_VERSION
            version_info.library_version = version.__version__

    def _is_alembic_supported(self) -> bool:
        with _create_scoped_session(self.scoped_session) as session:
            version_info = models.VersionInfoModel.find(session)

            if version_info is None:
                # `None` means this storage was created just now.
                return True

            return version_info.schema_version == models.SCHEMA_VERSION

    def _create_alembic_script(self) -> "alembic_script.ScriptDirectory":
        config = self._create_alembic_config()
        script = alembic_script.ScriptDirectory.from_config(config)
        return script

    def _create_alembic_config(self) -> "alembic_config.Config":
        alembic_dir = os.path.join(os.path.dirname(__file__), "alembic")

        config = alembic_config.Config(os.path.join(os.path.dirname(__file__), "alembic.ini"))
        config.set_main_option("script_location", escape_alembic_config_value(alembic_dir))
        config.set_main_option("sqlalchemy.url", escape_alembic_config_value(self.url))
        return config


def escape_alembic_config_value(value: str) -> str:
    # We must escape '%' in a value string because the character
    # is regarded as the trigger of variable expansion.
    # Please see the documentation of `configparser.BasicInterpolation` for more details.
    return value.replace("%", "%%")
