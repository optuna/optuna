from datetime import datetime
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from pfnopt.storages.rdb.models import BaseModel
from pfnopt.storages.rdb.models import StudyModel
from pfnopt.storages.rdb.models import TrialModel
from pfnopt.storages.rdb.models import VersionInfoModel
from pfnopt.structs import TrialState


@pytest.fixture
def session():
    # type: () -> Session

    engine = create_engine('sqlite:///:memory:')
    BaseModel.metadata.create_all(engine)
    return Session(bind=engine)


class TestTrialModel(object):

    @staticmethod
    def test_default_datetime(session):
        # type: (Session) -> None

        datetime_1 = datetime.now()

        session.add(TrialModel(state=TrialState.RUNNING))
        session.commit()

        datetime_2 = datetime.now()

        trial_model = session.query(TrialModel).first()
        assert datetime_1 < trial_model.datetime_start < datetime_2
        assert trial_model.datetime_complete is None

    @staticmethod
    def test_count(session):
        # type: (Session) -> None

        study_1 = StudyModel(study_id=1)
        study_2 = StudyModel(study_id=2)

        session.add(TrialModel(study_id=study_1.study_id, state=TrialState.COMPLETE))
        session.add(TrialModel(study_id=study_1.study_id, state=TrialState.RUNNING))
        session.add(TrialModel(study_id=study_2.study_id, state=TrialState.RUNNING))
        session.commit()

        assert 3 == TrialModel.count(session)
        assert 2 == TrialModel.count(session, study=study_1)
        assert 1 == TrialModel.count(session, state=TrialState.COMPLETE)


def test_version_info_model(session):
    # type: (Session) -> None

    session.add(VersionInfoModel(schema_version=1, library_version='0.0.1'))
    session.commit()

    # test check constraint of version_info_id
    session.add(VersionInfoModel(version_info_id=2, schema_version=2, library_version='0.0.2'))
    pytest.raises(IntegrityError, lambda: session.commit())
