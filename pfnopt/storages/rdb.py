from collections import defaultdict
from sqlalchemy import CheckConstraint
from sqlalchemy import Column
from sqlalchemy.engine import create_engine
from sqlalchemy import Enum
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import orm
from sqlalchemy import String
from sqlalchemy import UniqueConstraint
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA
import uuid

from pfnopt import distributions
from pfnopt import logging
from pfnopt.storages.base import BaseStorage
import pfnopt.trial as trial_module
from pfnopt.trial import State
from pfnopt import version

SCHEMA_VERSION = 1

Base = declarative_base()  # type: Any


class Study(Base):
    __tablename__ = 'studies'
    study_id = Column(Integer, primary_key=True)
    study_uuid = Column(String(255), unique=True)


class Trial(Base):
    __tablename__ = 'trials'
    trial_id = Column(Integer, primary_key=True)
    study_id = Column(Integer, ForeignKey('studies.study_id'))
    state = Column(Enum(State))
    value = Column(Float)
    system_attributes_json = Column(String(255))

    study = orm.relationship(Study)


class TrialParamDistribution(Base):
    __tablename__ = 'param_distributions'
    __table_args__ = (UniqueConstraint('trial_id', 'param_name'), )  # type: Any
    param_distribution_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey('trials.trial_id'))
    param_name = Column(String(255))
    distribution_json = Column(String(255))

    trial = orm.relationship(Trial)


# todo(sano): merge ParamDistribution and TrialParam because they are 1-to-1 relationship
class TrialParam(Base):
    __tablename__ = 'trial_params'
    __table_args__ = (UniqueConstraint('trial_id', 'param_distribution_id'), )  # type: Any
    trial_param_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey('trials.trial_id'))
    param_distribution_id = \
        Column(Integer, ForeignKey('param_distributions.param_distribution_id'))
    param_value = Column(Float)

    trial = orm.relationship(Trial)
    param_distribution = orm.relationship(TrialParamDistribution)


class TrialValue(Base):
    __tablename__ = 'trial_values'
    __table_args__ = (UniqueConstraint('trial_id', 'step'), )  # type: Any
    trial_value_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey('trials.trial_id'))
    step = Column(Integer)
    value = Column(Float)

    trial = orm.relationship(Trial)


class VersionInfo(Base):
    __tablename__ = 'version_info'
    # setting check constraint to ensure the number of rows is at most 1
    __table_args__ = (CheckConstraint('version_info_id=1'), )  # type: Any
    version_info_id = Column(Integer, primary_key=True, autoincrement=False, default=1)
    schema_version = Column(Integer)
    library_version = Column(String(255))


class RDBStorage(BaseStorage):

    def __init__(self, url, connect_args=None):
        # type: (str, Optional[Dict[str, Any]]) -> None

        connect_args = connect_args or {}
        self.engine = create_engine(url, connect_args=connect_args)
        self.scoped_session = orm.scoped_session(orm.sessionmaker(bind=self.engine))
        Base.metadata.create_all(self.engine)
        self._check_table_schema_compatibility()
        self.logger = logging.get_logger(__name__)

    def create_new_study_id(self):
        # type: () -> int

        session = self.scoped_session()

        while True:
            study_uuid = str(uuid.uuid4())
            study = session.query(Study).filter(Study.study_uuid == study_uuid).one_or_none()
            if study is None:
                break

        study = Study()
        study.study_uuid = study_uuid
        session.add(study)
        session.commit()

        return study.study_id

    def get_study_id_from_uuid(self, study_uuid):
        # type: (str) -> int

        session = self.scoped_session()
        study = session.query(Study).filter(Study.study_uuid == study_uuid).one_or_none()
        if study is None:
            raise ValueError('study_uuid {} does not exist.'.format(study_uuid))
        else:
            return study.study_id

    def get_study_uuid_from_id(self, study_id):
        # type: (int) -> str

        session = self.scoped_session()
        study = session.query(Study).filter(Study.study_id == study_id).one_or_none()
        if study is None:
            raise ValueError('study_id {} does not exist.'.format(study_id))
        else:
            return study.study_uuid

    def set_trial_param_distribution(self, trial_id, param_name, distribution):
        # type: (int, str, distributions.BaseDistribution) -> None

        session = self.scoped_session()
        trial = session.query(Trial).filter(Trial.trial_id == trial_id).one()

        # check if this distribution is compatible with previous ones in the same study
        param_distribution = session.query(TrialParamDistribution).join(Trial). \
            filter(Trial.study_id == trial.study_id). \
            filter(TrialParamDistribution.param_name == param_name).first()
        if param_distribution is not None:
            distribution_rdb = \
                distributions.json_to_distribution(param_distribution.distribution_json)
            distributions.check_distribution_compatibility(distribution_rdb, distribution)

        param_distribution = TrialParamDistribution()
        param_distribution.trial_id = trial_id
        param_distribution.param_name = param_name
        param_distribution.distribution_json = distributions.distribution_to_json(distribution)
        session.add(param_distribution)

        session.commit()

    def create_new_trial_id(self, study_id):
        # type: (int) -> int

        trial = Trial()
        trial.study_id = study_id
        trial.state = State.RUNNING

        system_attributes = \
            trial_module.SystemAttributes(datetime_start=None, datetime_complete=None)
        trial.system_attributes_json = trial_module.system_attrs_to_json(system_attributes)

        session = self.scoped_session()
        session.add(trial)
        session.commit()

        return trial.trial_id

    def set_trial_state(self, trial_id, state):
        # type: (int, trial_module.State) -> None

        session = self.scoped_session()
        trial = session.query(Trial).filter(Trial.trial_id == trial_id).one()

        trial.state = state
        session.commit()

    def set_trial_param(self, trial_id, param_name, param_value):
        # type: (int, str, float) -> None

        session = self.scoped_session()
        trial = session.query(Trial).filter(Trial.trial_id == trial_id).one()

        param_distribution = session.query(TrialParamDistribution). \
            filter(TrialParamDistribution.trial_id == trial.trial_id). \
            filter(TrialParamDistribution.param_name == param_name).one()

        # check if the parameter already exists
        trial_param = session.query(TrialParam). \
            filter(TrialParam.trial_id == trial_id). \
            filter(TrialParam.param_distribution.has(param_name=param_name)).one_or_none()
        if trial_param is not None:
            assert trial_param.param_value == param_value
            return

        trial_param = TrialParam()
        trial_param.trial_id = trial_id
        trial_param.param_distribution_id = param_distribution.param_distribution_id
        trial_param.param_value = param_value
        session.add(trial_param)

        try:
            session.commit()
        except IntegrityError as e:
            self.logger.debug(
                'Caught {}. This happens due to a known race condition. Another process/thread '
                'might have committed a record with the same unique key.'.format(repr(e)))
            session.rollback()

    def set_trial_value(self, trial_id, value):
        # type: (int, float) -> None

        session = self.scoped_session()
        trial = session.query(Trial).filter(Trial.trial_id == trial_id).one()
        trial.value = value
        session.commit()

    def set_trial_intermediate_value(self, trial_id, step, intermediate_value):
        # type: (int, int, float) -> None

        session = self.scoped_session()

        # the following line is to check that the specified trial_id exists in DB.
        session.query(Trial).filter(Trial.trial_id == trial_id).one()

        # check if the value at the same step already exists
        trial_value = session.query(TrialValue). \
            filter(TrialValue.trial_id == trial_id). \
            filter(TrialValue.step == step).one_or_none()
        if trial_value is not None:
            assert trial_value.value == intermediate_value
            return

        trial_value = TrialValue()
        trial_value.trial_id = trial_id
        trial_value.step = step
        trial_value.value = intermediate_value
        session.add(trial_value)

        try:
            session.commit()
        except IntegrityError as e:
            self.logger.debug(
                'Caught {}. This happens due to a known race condition. Another process/thread '
                'might have committed a record with the same unique key.'.format(repr(e)))
            session.rollback()

    def set_trial_system_attrs(self, trial_id, system_attrs):
        # type: (int, trial_module.SystemAttributes) -> None

        session = self.scoped_session()

        # the following line is to check that the specified trial_id exists in DB.
        trial = session.query(Trial).filter(Trial.trial_id == trial_id).one()

        trial.system_attributes_json = trial_module.system_attrs_to_json(system_attrs)
        session.commit()

    def get_trial(self, trial_id):
        # type: (int) -> trial_module.Trial

        session = self.scoped_session()
        trial = session.query(Trial).filter(Trial.trial_id == trial_id).one()
        params = session.query(TrialParam).filter(TrialParam.trial_id == trial_id).all()
        values = session.query(TrialValue).filter(TrialValue.trial_id == trial_id).all()

        return self._merge_trials_orm([trial], params, values)[0]

    def get_all_trials(self, study_id):
        # type: (int) -> List[trial_module.Trial]

        session = self.scoped_session()
        trials = session.query(Trial).filter(Trial.study_id == study_id).all()
        params = session.query(TrialParam).join(Trial). \
            filter(Trial.study_id == study_id).all()
        values = session.query(TrialValue).join(Trial). \
            filter(Trial.study_id == study_id).all()

        return self._merge_trials_orm(trials, params, values)

    @staticmethod
    def _merge_trials_orm(trials, trial_params, trial_intermediate_values):
        # type: (List[Trial], List[TrialParam], List[TrialValue]) -> List[trial_module.Trial]

        id_to_trial = {}
        for trial in trials:
            id_to_trial[trial.trial_id] = trial

        id_to_trial_params = defaultdict(list)  # type: Dict[int, List[TrialParam]]
        for param in trial_params:
            id_to_trial_params[param.trial_id].append(param)

        id_to_trial_intermediate_values = defaultdict(list)  # type: Dict[int, List[TrialValue]]
        for value in trial_intermediate_values:
            id_to_trial_intermediate_values[value.trial_id].append(value)

        result = []
        for trial_id, trial in id_to_trial.items():
            params = {}
            params_in_internal_repr = {}
            for param in id_to_trial_params[trial_id]:
                distribution = \
                    distributions.json_to_distribution(param.param_distribution.distribution_json)
                params[param.param_distribution.param_name] = \
                    distribution.to_external_repr(param.param_value)
                params_in_internal_repr[param.param_distribution.param_name] = param.param_value

            intermediate_values = {}
            for value in id_to_trial_intermediate_values[trial_id]:
                intermediate_values[value.step] = value.value

            result.append(trial_module.Trial(
                trial_id=trial_id,
                state=trial.state,
                params=params,
                system_attrs=trial_module.json_to_system_attrs(trial.system_attributes_json),
                user_attrs={},
                value=trial.value,
                intermediate_values=intermediate_values,
                params_in_internal_repr=params_in_internal_repr
            ))

        return result

    def _check_table_schema_compatibility(self):
        # type: () -> None

        session = self.scoped_session()

        version_info = session.query(VersionInfo).one_or_none()
        if version_info is None:
            version_info = VersionInfo()
            version_info.schema_version = SCHEMA_VERSION
            version_info.library_version = version.__version__
            session.add(version_info)
            try:
                session.commit()
            except IntegrityError as e:
                self.logger.debug(
                    'Ignoring {}. This happens due to a timing issue during initial setup of {} '
                    'table among multi threads/processes/nodes.'.format(
                        repr(e), VersionInfo.__tablename__))
                session.rollback()
        else:
            if version_info.schema_version != SCHEMA_VERSION:
                raise RuntimeError(
                    'The runtime pfnopt version {} is no longer compatible with the table schema '
                    '(set up by pfnopt {}).'.format(
                        version.__version__, version_info.library_version))

    def remove_session(self):
        # type: () -> None

        """Removes the current session.

        A session is stored in SQLALchemy's ThreadLocalRegistry for each thread. This method
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

        self.remove_session()
