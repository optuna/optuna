from typing import Any
from typing import List
from typing import Optional

from sqlalchemy import asc
from sqlalchemy import CheckConstraint
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import desc
from sqlalchemy import Enum
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import Integer
from sqlalchemy import orm
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base

from optuna import distributions
from optuna._study_direction import StudyDirection
from optuna.trial import TrialState


# Don't modify this version number anymore.
# The schema management functionality has been moved to alembic.
SCHEMA_VERSION = 12

MAX_INDEXED_STRING_LENGTH = 512
MAX_VERSION_LENGTH = 256

NOT_FOUND_MSG = "Record does not exist."

BaseModel: Any = declarative_base()


class StudyModel(BaseModel):
    __tablename__ = "studies"
    study_id = Column(Integer, primary_key=True)
    study_name = Column(String(MAX_INDEXED_STRING_LENGTH), index=True, unique=True, nullable=False)

    @classmethod
    def find_or_raise_by_id(
        cls, study_id: int, session: orm.Session, for_update: bool = False
    ) -> "StudyModel":

        query = session.query(cls).filter(cls.study_id == study_id)

        if for_update:
            query = query.with_for_update()

        study = query.one_or_none()
        if study is None:
            raise KeyError(NOT_FOUND_MSG)

        return study

    @classmethod
    def find_by_name(cls, study_name: str, session: orm.Session) -> Optional["StudyModel"]:

        study = session.query(cls).filter(cls.study_name == study_name).one_or_none()

        return study

    @classmethod
    def find_or_raise_by_name(cls, study_name: str, session: orm.Session) -> "StudyModel":

        study = cls.find_by_name(study_name, session)
        if study is None:
            raise KeyError(NOT_FOUND_MSG)

        return study


class StudyDirectionModel(BaseModel):
    __tablename__ = "study_directions"
    __table_args__: Any = (UniqueConstraint("study_id", "objective"),)
    study_direction_id = Column(Integer, primary_key=True)
    direction = Column(Enum(StudyDirection), nullable=False)
    study_id = Column(Integer, ForeignKey("studies.study_id"), nullable=False)
    objective = Column(Integer, nullable=False)

    study = orm.relationship(
        StudyModel, backref=orm.backref("directions", cascade="all, delete-orphan")
    )

    @classmethod
    def find_by_study_and_objective(
        cls, study: StudyModel, objective: int, session: orm.Session
    ) -> Optional["StudyDirectionModel"]:
        study_direction = (
            session.query(cls)
            .filter(cls.study_id == study.study_id)
            .filter(cls.objective == objective)
            .one_or_none()
        )

        return study_direction

    @classmethod
    def where_study_id(cls, study_id: int, session: orm.Session) -> List["StudyDirectionModel"]:

        return session.query(cls).filter(cls.study_id == study_id).all()


class StudyUserAttributeModel(BaseModel):
    __tablename__ = "study_user_attributes"
    __table_args__: Any = (UniqueConstraint("study_id", "key"),)
    study_user_attribute_id = Column(Integer, primary_key=True)
    study_id = Column(Integer, ForeignKey("studies.study_id"))
    key = Column(String(MAX_INDEXED_STRING_LENGTH))
    value_json = Column(Text())

    study = orm.relationship(
        StudyModel, backref=orm.backref("user_attributes", cascade="all, delete-orphan")
    )

    @classmethod
    def find_by_study_and_key(
        cls, study: StudyModel, key: str, session: orm.Session
    ) -> Optional["StudyUserAttributeModel"]:

        attribute = (
            session.query(cls)
            .filter(cls.study_id == study.study_id)
            .filter(cls.key == key)
            .one_or_none()
        )

        return attribute

    @classmethod
    def where_study_id(
        cls, study_id: int, session: orm.Session
    ) -> List["StudyUserAttributeModel"]:

        return session.query(cls).filter(cls.study_id == study_id).all()


class StudySystemAttributeModel(BaseModel):
    __tablename__ = "study_system_attributes"
    __table_args__: Any = (UniqueConstraint("study_id", "key"),)
    study_system_attribute_id = Column(Integer, primary_key=True)
    study_id = Column(Integer, ForeignKey("studies.study_id"))
    key = Column(String(MAX_INDEXED_STRING_LENGTH))
    value_json = Column(Text())

    study = orm.relationship(
        StudyModel, backref=orm.backref("system_attributes", cascade="all, delete-orphan")
    )

    @classmethod
    def find_by_study_and_key(
        cls, study: StudyModel, key: str, session: orm.Session
    ) -> Optional["StudySystemAttributeModel"]:

        attribute = (
            session.query(cls)
            .filter(cls.study_id == study.study_id)
            .filter(cls.key == key)
            .one_or_none()
        )

        return attribute

    @classmethod
    def where_study_id(
        cls, study_id: int, session: orm.Session
    ) -> List["StudySystemAttributeModel"]:

        return session.query(cls).filter(cls.study_id == study_id).all()


class TrialModel(BaseModel):
    __tablename__ = "trials"
    trial_id = Column(Integer, primary_key=True)
    # No `UniqueConstraint` is put on the `number` columns although it in practice is constrained
    # to be unique. This is to reduce code complexity as table-level locking would be required
    # otherwise. See https://github.com/optuna/optuna/pull/939#discussion_r387447632.
    number = Column(Integer)
    study_id = Column(Integer, ForeignKey("studies.study_id"))
    state = Column(Enum(TrialState), nullable=False)
    datetime_start = Column(DateTime)
    datetime_complete = Column(DateTime)

    study = orm.relationship(
        StudyModel, backref=orm.backref("trials", cascade="all, delete-orphan")
    )

    @classmethod
    def find_max_value_trial(
        cls, study_id: int, objective: int, session: orm.Session
    ) -> "TrialModel":

        trial = (
            session.query(cls)
            .filter(cls.study_id == study_id)
            .filter(cls.state == TrialState.COMPLETE)
            .join(TrialValueModel)
            .filter(TrialValueModel.objective == objective)
            .order_by(desc(TrialValueModel.value))
            .limit(1)
            .one_or_none()
        )
        if trial is None:
            raise ValueError(NOT_FOUND_MSG)
        return trial

    @classmethod
    def find_min_value_trial(
        cls, study_id: int, objective: int, session: orm.Session
    ) -> "TrialModel":

        trial = (
            session.query(cls)
            .filter(cls.study_id == study_id)
            .filter(cls.state == TrialState.COMPLETE)
            .join(TrialValueModel)
            .filter(TrialValueModel.objective == objective)
            .order_by(asc(TrialValueModel.value))
            .limit(1)
            .one_or_none()
        )
        if trial is None:
            raise ValueError(NOT_FOUND_MSG)
        return trial

    @classmethod
    def find_or_raise_by_id(
        cls, trial_id: int, session: orm.Session, for_update: bool = False
    ) -> "TrialModel":

        query = session.query(cls).filter(cls.trial_id == trial_id)

        # "FOR UPDATE" clause is used for row-level locking.
        # Please note that SQLite3 doesn't support this clause.
        if for_update:
            query = query.with_for_update()

        trial = query.one_or_none()
        if trial is None:
            raise KeyError(NOT_FOUND_MSG)

        return trial

    @classmethod
    def count(
        cls,
        session: orm.Session,
        study: Optional[StudyModel] = None,
        state: Optional[TrialState] = None,
    ) -> int:

        trial_count = session.query(func.count(cls.trial_id))
        if study is not None:
            trial_count = trial_count.filter(cls.study_id == study.study_id)
        if state is not None:
            trial_count = trial_count.filter(cls.state == state)

        return trial_count.scalar()

    def count_past_trials(self, session: orm.Session) -> int:

        trial_count = session.query(func.count(TrialModel.trial_id)).filter(
            TrialModel.study_id == self.study_id, TrialModel.trial_id < self.trial_id
        )
        return trial_count.scalar()


class TrialUserAttributeModel(BaseModel):
    __tablename__ = "trial_user_attributes"
    __table_args__: Any = (UniqueConstraint("trial_id", "key"),)
    trial_user_attribute_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trials.trial_id"))
    key = Column(String(MAX_INDEXED_STRING_LENGTH))
    value_json = Column(Text())

    trial = orm.relationship(
        TrialModel, backref=orm.backref("user_attributes", cascade="all, delete-orphan")
    )

    @classmethod
    def find_by_trial_and_key(
        cls, trial: TrialModel, key: str, session: orm.Session
    ) -> Optional["TrialUserAttributeModel"]:

        attribute = (
            session.query(cls)
            .filter(cls.trial_id == trial.trial_id)
            .filter(cls.key == key)
            .one_or_none()
        )

        return attribute

    @classmethod
    def where_trial_id(
        cls, trial_id: int, session: orm.Session
    ) -> List["TrialUserAttributeModel"]:

        return session.query(cls).filter(cls.trial_id == trial_id).all()


class TrialSystemAttributeModel(BaseModel):
    __tablename__ = "trial_system_attributes"
    __table_args__: Any = (UniqueConstraint("trial_id", "key"),)
    trial_system_attribute_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trials.trial_id"))
    key = Column(String(MAX_INDEXED_STRING_LENGTH))
    value_json = Column(Text())

    trial = orm.relationship(
        TrialModel, backref=orm.backref("system_attributes", cascade="all, delete-orphan")
    )

    @classmethod
    def find_by_trial_and_key(
        cls, trial: TrialModel, key: str, session: orm.Session
    ) -> Optional["TrialSystemAttributeModel"]:

        attribute = (
            session.query(cls)
            .filter(cls.trial_id == trial.trial_id)
            .filter(cls.key == key)
            .one_or_none()
        )

        return attribute

    @classmethod
    def where_trial_id(
        cls, trial_id: int, session: orm.Session
    ) -> List["TrialSystemAttributeModel"]:

        return session.query(cls).filter(cls.trial_id == trial_id).all()


class TrialParamModel(BaseModel):
    __tablename__ = "trial_params"
    __table_args__: Any = (UniqueConstraint("trial_id", "param_name"),)
    param_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trials.trial_id"))
    param_name = Column(String(MAX_INDEXED_STRING_LENGTH))
    param_value = Column(Float)
    distribution_json = Column(Text())

    trial = orm.relationship(
        TrialModel, backref=orm.backref("params", cascade="all, delete-orphan")
    )

    def check_and_add(self, session: orm.Session) -> None:

        self._check_compatibility_with_previous_trial_param_distributions(session)
        session.add(self)

    def _check_compatibility_with_previous_trial_param_distributions(
        self, session: orm.Session
    ) -> None:

        trial = TrialModel.find_or_raise_by_id(self.trial_id, session)

        previous_record = (
            session.query(TrialParamModel)
            .join(TrialModel)
            .filter(TrialModel.study_id == trial.study_id)
            .filter(TrialParamModel.param_name == self.param_name)
            .first()
        )
        if previous_record is not None:
            distributions.check_distribution_compatibility(
                distributions.json_to_distribution(previous_record.distribution_json),
                distributions.json_to_distribution(self.distribution_json),
            )

    @classmethod
    def find_by_trial_and_param_name(
        cls, trial: TrialModel, param_name: str, session: orm.Session
    ) -> Optional["TrialParamModel"]:

        param_distribution = (
            session.query(cls)
            .filter(cls.trial_id == trial.trial_id)
            .filter(cls.param_name == param_name)
            .one_or_none()
        )

        return param_distribution

    @classmethod
    def find_or_raise_by_trial_and_param_name(
        cls, trial: TrialModel, param_name: str, session: orm.Session
    ) -> "TrialParamModel":

        param_distribution = cls.find_by_trial_and_param_name(trial, param_name, session)

        if param_distribution is None:
            raise KeyError(NOT_FOUND_MSG)

        return param_distribution

    @classmethod
    def where_trial_id(cls, trial_id: int, session: orm.Session) -> List["TrialParamModel"]:

        trial_params = session.query(cls).filter(cls.trial_id == trial_id).all()

        return trial_params


class TrialValueModel(BaseModel):
    __tablename__ = "trial_values"
    __table_args__: Any = (UniqueConstraint("trial_id", "objective"),)
    trial_value_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trials.trial_id"), nullable=False)
    objective = Column(Integer, nullable=False)
    value = Column(Float, nullable=False)

    trial = orm.relationship(
        TrialModel, backref=orm.backref("values", cascade="all, delete-orphan")
    )

    @classmethod
    def find_by_trial_and_objective(
        cls, trial: TrialModel, objective: int, session: orm.Session
    ) -> Optional["TrialValueModel"]:

        trial_value = (
            session.query(cls)
            .filter(cls.trial_id == trial.trial_id)
            .filter(cls.objective == objective)
            .one_or_none()
        )

        return trial_value

    @classmethod
    def where_trial_id(cls, trial_id: int, session: orm.Session) -> List["TrialValueModel"]:

        trial_values = (
            session.query(cls).filter(cls.trial_id == trial_id).order_by(asc(cls.objective)).all()
        )

        return trial_values


class TrialIntermediateValueModel(BaseModel):
    __tablename__ = "trial_intermediate_values"
    __table_args__: Any = (UniqueConstraint("trial_id", "step"),)
    trial_intermediate_value_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trials.trial_id"), nullable=False)
    step = Column(Integer, nullable=False)
    intermediate_value = Column(Float, nullable=False)

    trial = orm.relationship(
        TrialModel, backref=orm.backref("intermediate_values", cascade="all, delete-orphan")
    )

    @classmethod
    def find_by_trial_and_step(
        cls, trial: TrialModel, step: int, session: orm.Session
    ) -> Optional["TrialIntermediateValueModel"]:

        trial_intermediate_value = (
            session.query(cls)
            .filter(cls.trial_id == trial.trial_id)
            .filter(cls.step == step)
            .one_or_none()
        )

        return trial_intermediate_value

    @classmethod
    def where_trial_id(
        cls, trial_id: int, session: orm.Session
    ) -> List["TrialIntermediateValueModel"]:

        trial_intermediate_values = session.query(cls).filter(cls.trial_id == trial_id).all()

        return trial_intermediate_values


class TrialHeartbeatModel(BaseModel):
    __tablename__ = "trial_heartbeats"
    __table_args__: Any = (UniqueConstraint("trial_id"),)
    trial_heartbeat_id = Column(Integer, primary_key=True)
    trial_id = Column(Integer, ForeignKey("trials.trial_id"), nullable=False)
    heartbeat = Column(DateTime, nullable=False, default=func.current_timestamp())

    trial = orm.relationship(
        TrialModel, backref=orm.backref("heartbeats", cascade="all, delete-orphan")
    )

    @classmethod
    def where_trial_id(
        cls, trial_id: int, session: orm.Session
    ) -> Optional["TrialHeartbeatModel"]:
        return session.query(cls).filter(cls.trial_id == trial_id).one_or_none()


class VersionInfoModel(BaseModel):
    __tablename__ = "version_info"
    # setting check constraint to ensure the number of rows is at most 1
    __table_args__: Any = (CheckConstraint("version_info_id=1"),)
    version_info_id = Column(Integer, primary_key=True, autoincrement=False, default=1)
    schema_version = Column(Integer)
    library_version = Column(String(MAX_VERSION_LENGTH))

    @classmethod
    def find(cls, session: orm.Session) -> "VersionInfoModel":

        version_info = session.query(cls).one_or_none()

        return version_info
