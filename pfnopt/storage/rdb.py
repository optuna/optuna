from sqlalchemy import Column
from sqlalchemy.engine import create_engine
from sqlalchemy import Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import orm
from sqlalchemy import String
from typing import Any  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA
import uuid

import pfnopt
from pfnopt import distributions
from pfnopt.storage.base import BaseStorage
import pfnopt.trial as trial_module
from pfnopt.trial import State

Base = declarative_base()  # type: Any
Session = orm.sessionmaker()


class Study(Base):
    __tablename__ = 'studies'
    study_id = Column(Integer, primary_key=True)
    study_uuid = Column(String(255), unique=True)


class StudyParam(Base):
    __tablename__ = 'study_params'
    study_param_id = Column(Integer, primary_key=True)
    study_id = Column(Integer, ForeignKey('studies.study_id'))
    param_name = Column(String(255))
    distribution_json = Column(String(255))

    study = orm.relationship(Study)


class Trial(Base):
    __tablename__ = 'trials'
    trial_id = Column(Integer, primary_key=True)
    study_id = Column(Integer, ForeignKey('studies.study_id'))
    state = Column(Enum(State))
    value = Column(Float)
    system_attributes_json = Column(String(255))

    study = orm.relationship(Study)


class TrialParam(Base):
    __tablename__ = 'trial_params'
    trial_param_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey('trials.trial_id'))
    study_param_id = Column(Integer, ForeignKey('study_params.study_param_id'))
    param_value = Column(Float)

    trial = orm.relationship(Trial)
    study_param = orm.relationship(StudyParam)


class TrialValue(Base):
    __tablename__ = 'trial_values'
    trial_value_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey('trials.trial_id'))
    step = Column(Integer)
    value = Column(Float)

    trial = orm.relationship(Trial)


class RDBStorage(BaseStorage):

    def __init__(self, url):
        # type: (str) -> None
        self.engine = create_engine(url)
        self.session = Session(bind=self.engine)
        Base.metadata.create_all(self.engine)

    def create_new_study_id(self):
        # type: () -> int
        while True:
            study_uuid = str(uuid.uuid4())
            study = self.session.query(Study).filter(Study.study_uuid == study_uuid).one_or_none()
            if study is None:
                break

        study = Study()
        study.study_uuid = study_uuid
        self.session.add(study)
        self.session.commit()

        return study.study_id

    def get_study_id_from_uuid(self, study_uuid):
        # type: (str) -> int
        study = self.session.query(Study).filter(Study.study_uuid == study_uuid).one_or_none()
        if study is None:
            raise ValueError('study_uuid {} does not exist.'.format(study_uuid))
        else:
            return study.study_id

    def get_study_uuid_from_id(self, study_id):
        # type: (int) -> str
        study = self.session.query(Study).filter(Study.study_id == study_id).one_or_none()
        if study is None:
            raise ValueError('study_id {} does not exist.'.format(study_id))
        else:
            return study.study_uuid

    def set_study_param_distribution(self, study_id, param_name, distribution):
        # type: (int, str, distributions.BaseDistribution) -> None
        # the following line is to check that the specified study_id exists in DB.
        self.session.query(Study).filter(Study.study_id == study_id).one()

        # check if the StudyParam already exists
        study_param = self.session.query(StudyParam). \
            filter(StudyParam.study_id == study_id). \
            filter(StudyParam.param_name == param_name).one_or_none()
        if study_param is not None:
            distribution_rdb = distributions.json_to_distribution(study_param.distribution_json)
            assert distribution_rdb == distribution
            return

        study_param = StudyParam()
        study_param.study_id = study_id
        study_param.param_name = param_name
        study_param.distribution_json = distributions.distribution_to_json(distribution)
        self.session.add(study_param)
        self.session.commit()

    def create_new_trial_id(self, study_id):
        # type: (int) -> int
        trial = Trial()

        trial.study_id = study_id
        trial.state = State.RUNNING

        system_attributes = \
            trial_module.SystemAttributes(datetime_start=None, datetime_complete=None)
        trial.system_attributes_json = trial_module.system_attrs_to_json(system_attributes)

        self.session.add(trial)
        self.session.commit()

        return trial.trial_id

    def set_trial_state(self, trial_id, state):
        # type: (int, trial_module.State) -> None
        trial = self.session.query(Trial).filter(Trial.trial_id == trial_id).one()

        trial.state = state
        self.session.commit()

    def set_trial_param(self, trial_id, param_name, param_value):
        # type: (int, str, float) -> None
        trial = self.session.query(Trial).filter(Trial.trial_id == trial_id).one()

        study_param = self.session.query(StudyParam). \
            filter(StudyParam.study_id == trial.study_id). \
            filter(StudyParam.param_name == param_name).one()

        # check if the parameter already exists
        trial_param = self.session.query(TrialParam). \
            filter(TrialParam.trial_id == trial_id). \
            filter(TrialParam.study_param.has(param_name=param_name)).one_or_none()
        if trial_param is not None:
            assert trial_param.param_value == param_value
            return

        trial_param = TrialParam()
        trial_param.trial_id = trial_id
        trial_param.study_param_id = study_param.study_param_id
        trial_param.param_value = param_value
        self.session.add(trial_param)

        self.session.commit()

    def set_trial_value(self, trial_id, value):
        # type: (int, float) -> None
        trial = self.session.query(Trial).filter(Trial.trial_id == trial_id).one()
        trial.value = value
        self.session.commit()

    def set_trial_intermediate_value(self, trial_id, step, intermediate_value):
        # type: (int, int, float) -> None
        # the following line is to check that the specified trial_id exists in DB.
        self.session.query(Trial).filter(Trial.trial_id == trial_id).one()

        # check if the value at the same step already exists
        trial_value = self.session.query(TrialValue). \
            filter(TrialValue.trial_id == trial_id). \
            filter(TrialValue.step == step).one_or_none()
        if trial_value is not None:
            assert trial_value.value == intermediate_value
            return

        trial_value = TrialValue()
        trial_value.trial_id = trial_id
        trial_value.step = step
        trial_value.value = intermediate_value
        self.session.add(trial_value)
        self.session.commit()

    def set_trial_system_attrs(self, trial_id, system_attrs):
        # type: (int, trial_module.SystemAttributes) -> None
        # the following line is to check that the specified trial_id exists in DB.
        trial = self.session.query(Trial).filter(Trial.trial_id == trial_id).one()

        trial.system_attributes_json = trial_module.system_attrs_to_json(system_attrs)
        self.session.commit()

    def get_trial(self, trial_id):
        # type: (int) -> trial_module.Trial
        trial = pfnopt.trial.Trial(trial_id)

        trial_rdb = self.session.query(Trial).filter(Trial.trial_id == trial_id).one()

        trial.value = trial_rdb.value
        trial.state = trial_rdb.state
        trial.system_attrs = trial_module.json_to_system_attrs(trial_rdb.system_attributes_json)

        trial_params = self.session.query(TrialParam).filter(TrialParam.trial_id == trial_id).all()
        for param in trial_params:
            distribution = \
                distributions.json_to_distribution(param.study_param.distribution_json)
            trial.params[param.study_param.param_name] = \
                distribution.to_external_repr(param.param_value)
            trial.params_in_internal_repr[param.study_param.param_name] = param.param_value

        trial_intermediate_values = self.session.query(TrialValue). \
            filter(TrialValue.trial_id == trial_id).all()
        for iv in trial_intermediate_values:
            trial.intermediate_values[iv.step] = iv.value

        return trial

    def get_all_trials(self, study_id):
        # type: (int) -> List[trial_module.Trial]
        trials = self.session.query(Trial). \
            filter(Trial.study_id == study_id).all()

        return [self.get_trial(t.trial_id) for t in trials]

    def close(self):
        # type: () -> None
        self.session.close()
