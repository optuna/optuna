from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from optuna.storages.rdb.models import BaseModel
from optuna.storages.rdb.models import StudyModel
from optuna.storages.rdb.models import StudySystemAttributeModel
from optuna.storages.rdb.models import TrialModel
from optuna.storages.rdb.models import TrialSystemAttributeModel
from optuna.storages.rdb.models import TrialUserAttributeModel
from optuna.storages.rdb.models import VersionInfoModel
from optuna.study import StudyDirection
from optuna.trial import TrialState


@pytest.fixture
def session():
    # type: () -> Session

    engine = create_engine("sqlite:///:memory:")
    BaseModel.metadata.create_all(engine)
    return Session(bind=engine)


class TestStudySystemAttributeModel(object):
    @staticmethod
    def test_find_by_study_and_key(session):
        # type: (Session) -> None

        study = StudyModel(study_id=1, study_name="test-study")
        session.add(
            StudySystemAttributeModel(study_id=study.study_id, key="sample-key", value_json="1")
        )
        session.commit()

        attr = StudySystemAttributeModel.find_by_study_and_key(study, "sample-key", session)
        assert attr is not None and "1" == attr.value_json

        assert StudySystemAttributeModel.find_by_study_and_key(study, "not-found", session) is None

    @staticmethod
    def test_where_study_id(session):
        # type: (Session) -> None

        sample_study = StudyModel(study_id=1, study_name="test-study")
        empty_study = StudyModel(study_id=2, study_name="test-study")

        session.add(
            StudySystemAttributeModel(
                study_id=sample_study.study_id, key="sample-key", value_json="1"
            )
        )

        assert 1 == len(StudySystemAttributeModel.where_study_id(sample_study.study_id, session))
        assert 0 == len(StudySystemAttributeModel.where_study_id(empty_study.study_id, session))
        # Check the case of unknown study_id.
        assert 0 == len(StudySystemAttributeModel.where_study_id(-1, session))

    @staticmethod
    def test_cascade_delete_on_study(session):
        # type: (Session) -> None

        study_id = 1
        study = StudyModel(
            study_id=study_id, study_name="test-study", direction=StudyDirection.MINIMIZE
        )
        study.system_attributes.append(
            StudySystemAttributeModel(study_id=study_id, key="sample-key1", value_json="1")
        )
        study.system_attributes.append(
            StudySystemAttributeModel(study_id=study_id, key="sample-key2", value_json="2")
        )
        session.add(study)
        session.commit()

        assert 2 == len(StudySystemAttributeModel.where_study_id(study_id, session))

        session.delete(study)
        session.commit()

        assert 0 == len(StudySystemAttributeModel.where_study_id(study_id, session))


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

        study_1 = StudyModel(study_id=1, study_name="test-study-1")
        study_2 = StudyModel(study_id=2, study_name="test-study-2")

        session.add(TrialModel(study_id=study_1.study_id, state=TrialState.COMPLETE))
        session.add(TrialModel(study_id=study_1.study_id, state=TrialState.RUNNING))
        session.add(TrialModel(study_id=study_2.study_id, state=TrialState.RUNNING))
        session.commit()

        assert 3 == TrialModel.count(session)
        assert 2 == TrialModel.count(session, study=study_1)
        assert 1 == TrialModel.count(session, state=TrialState.COMPLETE)

    @staticmethod
    def test_count_past_trials(session):
        # type: (Session) -> None

        study_1 = StudyModel(study_id=1, study_name="test-study-1")
        study_2 = StudyModel(study_id=2, study_name="test-study-2")

        trial_1_1 = TrialModel(study_id=study_1.study_id, state=TrialState.COMPLETE)
        session.add(trial_1_1)
        session.commit()
        assert 0 == trial_1_1.count_past_trials(session)

        trial_1_2 = TrialModel(study_id=study_1.study_id, state=TrialState.RUNNING)
        session.add(trial_1_2)
        session.commit()
        assert 1 == trial_1_2.count_past_trials(session)

        trial_2_1 = TrialModel(study_id=study_2.study_id, state=TrialState.RUNNING)
        session.add(trial_2_1)
        session.commit()
        assert 0 == trial_2_1.count_past_trials(session)

    @staticmethod
    def test_cascade_delete_on_study(session):
        # type: (Session) -> None

        study_id = 1
        study = StudyModel(
            study_id=study_id, study_name="test-study", direction=StudyDirection.MINIMIZE
        )
        study.trials.append(TrialModel(study_id=study.study_id, state=TrialState.COMPLETE))
        study.trials.append(TrialModel(study_id=study.study_id, state=TrialState.RUNNING))
        session.add(study)
        session.commit()

        assert 2 == len(TrialModel.where_study(study, session))

        session.delete(study)
        session.commit()

        assert 0 == len(TrialModel.where_study(study, session))


class TestTrialUserAttributeModel(object):
    @staticmethod
    def test_find_by_trial_and_key(session):
        # type: (Session) -> None

        study = StudyModel(study_id=1, study_name="test-study")
        trial = TrialModel(study_id=study.study_id)

        session.add(
            TrialUserAttributeModel(trial_id=trial.trial_id, key="sample-key", value_json="1")
        )
        session.commit()

        attr = TrialUserAttributeModel.find_by_trial_and_key(trial, "sample-key", session)
        assert attr is not None
        assert "1" == attr.value_json
        assert TrialUserAttributeModel.find_by_trial_and_key(trial, "not-found", session) is None

    @staticmethod
    def test_where_study(session):
        # type: (Session) -> None

        study = StudyModel(study_id=1, study_name="test-study", direction=StudyDirection.MINIMIZE)
        trial = TrialModel(trial_id=1, study_id=study.study_id, state=TrialState.COMPLETE)

        session.add(study)
        session.add(trial)
        session.add(
            TrialUserAttributeModel(trial_id=trial.trial_id, key="sample-key", value_json="1")
        )
        session.commit()

        user_attributes = TrialUserAttributeModel.where_study(study, session)
        assert 1 == len(user_attributes)
        assert "sample-key" == user_attributes[0].key
        assert "1" == user_attributes[0].value_json

    @staticmethod
    def test_where_trial(session):
        # type: (Session) -> None

        study = StudyModel(study_id=1, study_name="test-study", direction=StudyDirection.MINIMIZE)
        trial = TrialModel(trial_id=1, study_id=study.study_id, state=TrialState.COMPLETE)

        session.add(
            TrialUserAttributeModel(trial_id=trial.trial_id, key="sample-key", value_json="1")
        )
        session.commit()

        user_attributes = TrialUserAttributeModel.where_trial(trial, session)
        assert 1 == len(user_attributes)
        assert "sample-key" == user_attributes[0].key
        assert "1" == user_attributes[0].value_json

    @staticmethod
    def test_all(session):
        # type: (Session) -> None

        study = StudyModel(study_id=1, study_name="test-study", direction=StudyDirection.MINIMIZE)
        trial = TrialModel(trial_id=1, study_id=study.study_id, state=TrialState.COMPLETE)

        session.add(
            TrialUserAttributeModel(trial_id=trial.trial_id, key="sample-key", value_json="1")
        )
        session.commit()

        user_attributes = TrialUserAttributeModel.all(session)
        assert 1 == len(user_attributes)
        assert "sample-key" == user_attributes[0].key
        assert "1" == user_attributes[0].value_json

    @staticmethod
    def test_cascade_delete_on_trial(session):
        # type: (Session) -> None

        trial_id = 1
        study = StudyModel(study_id=1, study_name="test-study", direction=StudyDirection.MINIMIZE)
        trial = TrialModel(trial_id=trial_id, study_id=study.study_id, state=TrialState.COMPLETE)
        trial.user_attributes.append(
            TrialUserAttributeModel(trial_id=trial_id, key="sample-key1", value_json="1")
        )
        trial.user_attributes.append(
            TrialUserAttributeModel(trial_id=trial_id, key="sample-key2", value_json="2")
        )
        study.trials.append(trial)
        session.add(study)
        session.commit()

        assert 2 == len(TrialUserAttributeModel.where_trial_id(trial_id, session))

        session.delete(trial)
        session.commit()

        assert 0 == len(TrialUserAttributeModel.where_trial_id(trial_id, session))


class TestTrialSystemAttributeModel(object):
    @staticmethod
    def test_find_by_trial_and_key(session):
        # type: (Session) -> None

        study = StudyModel(study_id=1, study_name="test-study")
        trial = TrialModel(study_id=study.study_id)

        session.add(
            TrialSystemAttributeModel(trial_id=trial.trial_id, key="sample-key", value_json="1")
        )
        session.commit()

        attr = TrialSystemAttributeModel.find_by_trial_and_key(trial, "sample-key", session)
        assert attr is not None
        assert "1" == attr.value_json
        assert TrialSystemAttributeModel.find_by_trial_and_key(trial, "not-found", session) is None

    @staticmethod
    def test_where_study(session):
        # type: (Session) -> None

        study = StudyModel(study_id=1, study_name="test-study", direction=StudyDirection.MINIMIZE)
        trial = TrialModel(trial_id=1, study_id=study.study_id, state=TrialState.COMPLETE)

        session.add(study)
        session.add(trial)
        session.add(
            TrialSystemAttributeModel(trial_id=trial.trial_id, key="sample-key", value_json="1")
        )
        session.commit()

        system_attributes = TrialSystemAttributeModel.where_study(study, session)
        assert 1 == len(system_attributes)
        assert "sample-key" == system_attributes[0].key
        assert "1" == system_attributes[0].value_json

    @staticmethod
    def test_where_trial(session):
        # type: (Session) -> None

        study = StudyModel(study_id=1, study_name="test-study", direction=StudyDirection.MINIMIZE)
        trial = TrialModel(trial_id=1, study_id=study.study_id, state=TrialState.COMPLETE)

        session.add(
            TrialSystemAttributeModel(trial_id=trial.trial_id, key="sample-key", value_json="1")
        )
        session.commit()

        system_attributes = TrialSystemAttributeModel.where_trial(trial, session)
        assert 1 == len(system_attributes)
        assert "sample-key" == system_attributes[0].key
        assert "1" == system_attributes[0].value_json

    @staticmethod
    def test_all(session):
        # type: (Session) -> None

        study = StudyModel(study_id=1, study_name="test-study", direction=StudyDirection.MINIMIZE)
        trial = TrialModel(trial_id=1, study_id=study.study_id, state=TrialState.COMPLETE)

        session.add(
            TrialSystemAttributeModel(trial_id=trial.trial_id, key="sample-key", value_json="1")
        )
        session.commit()

        system_attributes = TrialSystemAttributeModel.all(session)
        assert 1 == len(system_attributes)
        assert "sample-key" == system_attributes[0].key
        assert "1" == system_attributes[0].value_json

    @staticmethod
    def test_cascade_delete_on_trial(session):
        # type: (Session) -> None

        trial_id = 1
        study = StudyModel(study_id=1, study_name="test-study", direction=StudyDirection.MINIMIZE)
        trial = TrialModel(trial_id=trial_id, study_id=study.study_id, state=TrialState.COMPLETE)
        trial.system_attributes.append(
            TrialSystemAttributeModel(trial_id=trial_id, key="sample-key1", value_json="1")
        )
        trial.system_attributes.append(
            TrialSystemAttributeModel(trial_id=trial_id, key="sample-key2", value_json="2")
        )
        study.trials.append(trial)
        session.add(study)
        session.commit()

        assert 2 == len(TrialSystemAttributeModel.where_trial_id(trial_id, session))

        session.delete(trial)
        session.commit()

        assert 0 == len(TrialSystemAttributeModel.where_trial_id(trial_id, session))


class TestVersionInfoModel(object):
    @staticmethod
    def test_version_info_id_constraint(session):
        # type: (Session) -> None

        session.add(VersionInfoModel(schema_version=1, library_version="0.0.1"))
        session.commit()

        # Test check constraint of version_info_id.
        session.add(VersionInfoModel(version_info_id=2, schema_version=2, library_version="0.0.2"))
        pytest.raises(IntegrityError, lambda: session.commit())
