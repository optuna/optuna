from datetime import datetime
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from pfnopt.frozen_trial import State
from pfnopt.storages.rdb.models import BaseModel
from pfnopt.storages.rdb.models import TrialModel
from pfnopt.storages.rdb.models import VersionInfoModel


def test_trial_model():
    # type: () -> None

    engine = create_engine('sqlite:///:memory:')
    session = Session(bind=engine)
    BaseModel.metadata.create_all(engine)

    datetime_1 = datetime.now()

    session.add(TrialModel(state=State.RUNNING))
    session.commit()

    datetime_2 = datetime.now()

    trial_model = session.query(TrialModel).first()
    assert datetime_1 < trial_model.datetime_start < datetime_2
    assert trial_model.datetime_complete is None


def test_version_info_model():
    # type: () -> None

    engine = create_engine('sqlite:///:memory:')
    session = Session(bind=engine)
    BaseModel.metadata.create_all(engine)

    session.add(VersionInfoModel(schema_version=1, library_version='0.0.1'))
    session.commit()

    # test check constraint of version_info_id
    session.add(VersionInfoModel(version_info_id=2, schema_version=2, library_version='0.0.2'))
    pytest.raises(IntegrityError, lambda: session.commit())
