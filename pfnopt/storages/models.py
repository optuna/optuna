from datetime import datetime

from pfnopt import distributions, version
from sqlalchemy import CheckConstraint
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import orm
from sqlalchemy import String
from sqlalchemy import UniqueConstraint
from typing import Any, Optional, List  # NOQA
from pfnopt.trial import State
from sqlalchemy.orm import Session

SCHEMA_VERSION = 4

BaseModel = declarative_base()  # type: Any


class StudyModel(BaseModel):
    __tablename__ = 'studies'
    study_id = Column(Integer, primary_key=True)
    study_uuid = Column(String(255), unique=True)

    @classmethod
    def find_by_id(cls, study_id, session, allow_none=True):
        # type: (int, Session, bool) -> Optional[StudyModel]

        study = session.query(cls).filter(cls.study_id == study_id).one_or_none()
        if study is None and not allow_none:
            raise ValueError('study_id {} does not exist.'.format(study_id))

        return study

    @classmethod
    def find_by_uuid(cls, study_uuid, session, allow_none=True):
        # type: (str, Session, bool) -> Optional[StudyModel]

        study = session.query(cls).filter(cls.study_uuid == study_uuid).one_or_none()
        if study is None and not allow_none:
            raise ValueError('study_uuid {} does not exist.'.format(study_uuid))

        return study


class StudyUserAttributeModel(BaseModel):
    __tablename__ = 'study_user_attributes'
    __table_args__ = (UniqueConstraint('study_id', 'key'), )  # type: Any
    study_user_attribute_id = Column(Integer, primary_key=True)
    study_id = Column(Integer, ForeignKey('studies.study_id'))
    key = Column(String(255))
    value_json = Column(String(255))

    study = orm.relationship(StudyModel)

    @classmethod
    def find_by_study_id_and_key(cls, study_id, key, session):
        # type: (int, str, Session) -> Optional[StudyUserAttributeModel]

        attribute = session.query(cls). \
            filter(cls.study_id == study_id).filter(cls.key == key).one_or_none()

        return attribute

    @classmethod
    def where_study_id(cls, study_id, session):
        # type: (int, Session) -> List[StudyUserAttributeModel]

        return session.query(cls).filter(cls.study_id == study_id).all()


class TrialModel(BaseModel):
    __tablename__ = 'trials'
    trial_id = Column(Integer, primary_key=True)
    study_id = Column(Integer, ForeignKey('studies.study_id'))
    state = Column(Enum(State))
    value = Column(Float)
    user_attributes_json = Column(String(255))
    datetime_start = Column(DateTime, default=datetime.now)
    datetime_complete = Column(DateTime)

    study = orm.relationship(StudyModel)

    @classmethod
    def find_by_id(cls, trial_id, session, allow_none=True):
        # type: (str, trial_id, Session, bool) -> Optional[TrialModel]

        trial = session.query(cls).filter(cls.trial_id == trial_id).one_or_none()
        if trial is None and not allow_none:
            raise ValueError('trial_id {} does not exist.'.format(trial_id))

        return trial

    @classmethod
    def where_study(cls, study, session):
        # type: (StudyModel, Session) -> List[TrialModel]

        trials = session.query(cls).filter(cls.study_id == study.study_id).all()

        return trials


class TrialParamDistributionModel(BaseModel):
    __tablename__ = 'param_distributions'
    __table_args__ = (UniqueConstraint('trial_id', 'param_name'), )  # type: Any
    param_distribution_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey('trials.trial_id'))
    param_name = Column(String(255))
    distribution_json = Column(String(255))

    trial = orm.relationship(TrialModel)

    def check_and_add(self, session):
        # type: (Session) -> None

        self._check_compatibility_with_previous_trial_param_distributions(session)
        session.add(self)

    def _check_compatibility_with_previous_trial_param_distributions(self, session):
        # type: (Session) -> None

        trial = TrialModel.find_by_id(self.trial_id, session, allow_none=False)

        previous_record = session.query(self.__class__).join(TrialModel). \
            filter(TrialModel.study_id == trial.study_id). \
            filter(self.__class__.param_name == self.param_name).first()
        if previous_record is not None:
            distributions.check_distribution_compatibility(
                distributions.json_to_distribution(previous_record.distribution_json),
                distributions.json_to_distribution(self.distribution_json))

    @classmethod
    def find_by_trial_and_param_name(cls, trial, param_name, session, allow_none=True):
        # type: (TrialModel, str, Session) -> Optional[TrialParamDistributionModel]

        param_distribution = session.query(cls). \
            filter(cls.trial_id == trial.trial_id). \
            filter(cls.param_name == param_name).one_or_none()

        if param_distribution is None and not allow_none:
            raise ValueError(
                'Record does not exist with trial_id {} and param_name {}.'.format(
                    trial.trial_id, param_name
                )
            )

        return param_distribution


# todo(sano): merge ParamDistribution and TrialParam because they are 1-to-1 relationship
class TrialParamModel(BaseModel):
    __tablename__ = 'trial_params'
    __table_args__ = (UniqueConstraint('trial_id', 'param_distribution_id'), )  # type: Any
    trial_param_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey('trials.trial_id'))
    param_distribution_id = \
        Column(Integer, ForeignKey('param_distributions.param_distribution_id'))
    param_value = Column(Float)

    trial = orm.relationship(TrialModel)
    param_distribution = orm.relationship(TrialParamDistributionModel)

    @classmethod
    def find_by_trial_and_param_name(cls, trial, param_name, session):
        # type: (TrialModel, str, Session) -> Optional[TrialParamModel]

        trial_param = session.query(cls). \
            filter(cls.trial_id == trial.trial_id). \
            filter(cls.param_distribution.has(param_name=param_name)).one_or_none()

        return trial_param

    @classmethod
    def where_study(cls, study, session):
        # type: (StudyModel, Session) -> List[TrialParamModel]

        trial_params = session.query(cls).join(TrialModel). \
            filter(TrialModel.study_id == study.study_id).all()

        return trial_params


    @classmethod
    def where_trial(cls, trial, session):
        # type: (TrialModel, Session) -> List[TrialParamModel]

        trial_params = session.query(cls).filter(cls.trial_id == trial.trial_id).all()

        return trial_params


class TrialValueModel(BaseModel):
    __tablename__ = 'trial_values'
    __table_args__ = (UniqueConstraint('trial_id', 'step'), )  # type: Any
    trial_value_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey('trials.trial_id'))
    step = Column(Integer)
    value = Column(Float)

    trial = orm.relationship(TrialModel)

    @classmethod
    def find_by_trial_and_step(cls, trial, step, session):
        # type: (TrialModel, int, Session) -> Optional[TrialValueModel]

        trial_value = session.query(cls). \
            filter(cls.trial_id == trial.trial_id). \
            filter(cls.step == step).one_or_none()

        return trial_value

    @classmethod
    def where_study(cls, study, session):
        # type: (StudyModel, Session) -> List[TrialValueModel]

        trial_values = session.query(cls).join(TrialModel). \
            filter(TrialModel.study_id == study.study_id).all()

        return trial_values

    @classmethod
    def where_trial(cls, trial, session):
        # type: (TrialModel, Session) -> List[TrialValueModel]

        trial_values = session.query(cls).filter(cls.trial_id == trial.trial_id).all()

        return trial_values


class VersionInfoModel(BaseModel):
    __tablename__ = 'version_info'
    # setting check constraint to ensure the number of rows is at most 1
    __table_args__ = (CheckConstraint('version_info_id=1'), )  # type: Any
    version_info_id = Column(Integer, primary_key=True, autoincrement=False, default=1)
    schema_version = Column(Integer)
    library_version = Column(String(255))

    @classmethod
    def find_or_create(cls, session):
        # type: (Session) -> VersionInfoModel

        version_info = session.query(cls).one_or_none()
        if version_info is None:
            version_info = VersionInfoModel(
                schema_version=SCHEMA_VERSION,
                library_version=version.__version__
            )
            session.add(version_info)

        return version_info
