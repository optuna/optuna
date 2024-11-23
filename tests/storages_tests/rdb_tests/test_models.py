from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from optuna.storages._rdb.models import BaseModel
from optuna.storages._rdb.models import StudyDirectionModel
from optuna.storages._rdb.models import StudyModel
from optuna.storages._rdb.models import StudySystemAttributeModel
from optuna.storages._rdb.models import TrialHeartbeatModel
from optuna.storages._rdb.models import TrialIntermediateValueModel
from optuna.storages._rdb.models import TrialModel
from optuna.storages._rdb.models import TrialSystemAttributeModel
from optuna.storages._rdb.models import TrialUserAttributeModel
from optuna.storages._rdb.models import TrialValueModel
from optuna.storages._rdb.models import VersionInfoModel
from optuna.study._study_direction import StudyDirection
from optuna.trial import TrialState


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    BaseModel.metadata.create_all(engine)
    return Session(bind=engine)


class TestStudyDirectionModel:
    @staticmethod
    def _create_model(session: Session) -> StudyModel:
        study = StudyModel(study_id=1, study_name="test-study")
        dummy_study = StudyModel(study_id=2, study_name="dummy-study")
        session.add(
            StudyDirectionModel(
                study_id=study.study_id, direction=StudyDirection.MINIMIZE, objective=0
            )
        )
        session.add(
            StudyDirectionModel(
                study_id=dummy_study.study_id, direction=StudyDirection.MINIMIZE, objective=0
            )
        )
        session.commit()
        return study

    @staticmethod
    def test_where_study_id(session: Session) -> None:
        study = TestStudyDirectionModel._create_model(session)
        assert 1 == len(StudyDirectionModel.where_study_id(study.study_id, session))
        assert 0 == len(StudyDirectionModel.where_study_id(-1, session))

    @staticmethod
    def test_cascade_delete_on_study(session: Session) -> None:
        directions = [
            StudyDirectionModel(study_id=1, direction=StudyDirection.MINIMIZE, objective=0),
            StudyDirectionModel(study_id=1, direction=StudyDirection.MAXIMIZE, objective=1),
        ]
        study = StudyModel(study_id=1, study_name="test-study", directions=directions)
        session.add(study)
        session.commit()

        assert 2 == len(StudyDirectionModel.where_study_id(study.study_id, session))

        session.delete(study)
        session.commit()

        assert 0 == len(StudyDirectionModel.where_study_id(study.study_id, session))


class TestStudySystemAttributeModel:
    @staticmethod
    def test_find_by_study_and_key(session: Session) -> None:
        study = StudyModel(study_id=1, study_name="test-study")
        session.add(
            StudySystemAttributeModel(study_id=study.study_id, key="sample-key", value_json="1")
        )
        session.commit()

        attr = StudySystemAttributeModel.find_by_study_and_key(study, "sample-key", session)
        assert attr is not None and "1" == attr.value_json

        assert StudySystemAttributeModel.find_by_study_and_key(study, "not-found", session) is None

    @staticmethod
    def test_where_study_id(session: Session) -> None:
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
    def test_cascade_delete_on_study(session: Session) -> None:
        study_id = 1
        direction = StudyDirectionModel(direction=StudyDirection.MINIMIZE, objective=0)
        study = StudyModel(study_id=study_id, study_name="test-study", directions=[direction])
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


class TestTrialModel:
    @staticmethod
    def test_default_datetime(session: Session) -> None:
        # Regardless of the initial state the trial created here should have null datetime_start
        session.add(TrialModel(state=TrialState.WAITING))
        session.commit()

        trial_model = session.query(TrialModel).first()

        assert trial_model is not None
        assert trial_model.datetime_start is None
        assert trial_model.datetime_complete is None

    @staticmethod
    def test_count(session: Session) -> None:
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
    def test_count_past_trials(session: Session) -> None:
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
    def test_cascade_delete_on_study(session: Session) -> None:
        study_id = 1
        direction = StudyDirectionModel(direction=StudyDirection.MINIMIZE, objective=0)
        study = StudyModel(study_id=study_id, study_name="test-study", directions=[direction])
        study.trials.append(TrialModel(study_id=study.study_id, state=TrialState.COMPLETE))
        study.trials.append(TrialModel(study_id=study.study_id, state=TrialState.RUNNING))
        session.add(study)
        session.commit()

        assert 2 == TrialModel.count(session, study)

        session.delete(study)
        session.commit()

        assert 0 == TrialModel.count(session, study)


class TestTrialUserAttributeModel:
    @staticmethod
    def test_find_by_trial_and_key(session: Session) -> None:
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
    def test_cascade_delete_on_trial(session: Session) -> None:
        trial_id = 1
        direction = StudyDirectionModel(direction=StudyDirection.MINIMIZE, objective=0)
        study = StudyModel(study_id=1, study_name="test-study", directions=[direction])
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


class TestTrialSystemAttributeModel:
    @staticmethod
    def test_find_by_trial_and_key(session: Session) -> None:
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
    def test_cascade_delete_on_trial(session: Session) -> None:
        trial_id = 1
        direction = StudyDirectionModel(direction=StudyDirection.MINIMIZE, objective=0)
        study = StudyModel(study_id=1, study_name="test-study", directions=[direction])
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


class TestTrialValueModel:
    @staticmethod
    def _create_model(session: Session) -> TrialModel:
        direction = StudyDirectionModel(direction=StudyDirection.MINIMIZE, objective=0)
        study = StudyModel(study_id=1, study_name="test-study", directions=[direction])
        trial = TrialModel(trial_id=1, study_id=study.study_id, state=TrialState.COMPLETE)
        session.add(study)
        session.add(trial)
        session.add(
            TrialValueModel(
                trial_id=trial.trial_id,
                objective=0,
                value=10,
                value_type=TrialValueModel.TrialValueType.FINITE,
            )
        )
        session.commit()
        return trial

    @staticmethod
    def test_find_by_trial_and_objective(session: Session) -> None:
        trial = TestTrialValueModel._create_model(session)
        trial_value = TrialValueModel.find_by_trial_and_objective(trial, 0, session)
        assert trial_value is not None
        assert 10 == trial_value.value
        assert TrialValueModel.find_by_trial_and_objective(trial, 1, session) is None

    @staticmethod
    def test_where_trial_id(session: Session) -> None:
        trial = TestTrialValueModel._create_model(session)
        trial_values = TrialValueModel.where_trial_id(trial.trial_id, session)
        assert 1 == len(trial_values)
        assert 0 == trial_values[0].objective
        assert 10 == trial_values[0].value

    @staticmethod
    def test_cascade_delete_on_trial(session: Session) -> None:
        trial = TestTrialValueModel._create_model(session)
        trial.values.append(
            TrialValueModel(
                trial_id=1, objective=1, value=20, value_type=TrialValueModel.TrialValueType.FINITE
            )
        )
        session.commit()

        assert 2 == len(TrialValueModel.where_trial_id(trial.trial_id, session))

        session.delete(trial)
        session.commit()

        assert 0 == len(TrialValueModel.where_trial_id(trial.trial_id, session))


class TestTrialIntermediateValueModel:
    @staticmethod
    def _create_model(session: Session) -> TrialModel:
        direction = StudyDirectionModel(direction=StudyDirection.MINIMIZE, objective=0)
        study = StudyModel(study_id=1, study_name="test-study", directions=[direction])
        trial = TrialModel(trial_id=1, study_id=study.study_id, state=TrialState.COMPLETE)
        session.add(study)
        session.add(trial)
        session.add(
            TrialIntermediateValueModel(
                trial_id=trial.trial_id,
                step=0,
                intermediate_value=10,
                intermediate_value_type=TrialIntermediateValueModel.TrialIntermediateValueType.FINITE,  # noqa: E501
            )
        )
        session.commit()
        return trial

    @staticmethod
    def test_find_by_trial_and_step(session: Session) -> None:
        trial = TestTrialIntermediateValueModel._create_model(session)
        trial_intermediate_value = TrialIntermediateValueModel.find_by_trial_and_step(
            trial, 0, session
        )
        assert trial_intermediate_value is not None
        assert 10 == trial_intermediate_value.intermediate_value
        assert TrialIntermediateValueModel.find_by_trial_and_step(trial, 1, session) is None

    @staticmethod
    def test_where_trial_id(session: Session) -> None:
        trial = TestTrialIntermediateValueModel._create_model(session)
        trial_intermediate_values = TrialIntermediateValueModel.where_trial_id(
            trial.trial_id, session
        )
        assert 1 == len(trial_intermediate_values)
        assert 0 == trial_intermediate_values[0].step
        assert 10 == trial_intermediate_values[0].intermediate_value

    @staticmethod
    def test_cascade_delete_on_trial(session: Session) -> None:
        trial = TestTrialIntermediateValueModel._create_model(session)
        trial.intermediate_values.append(
            TrialIntermediateValueModel(
                trial_id=1,
                step=1,
                intermediate_value=20,
                intermediate_value_type=TrialIntermediateValueModel.TrialIntermediateValueType.FINITE,  # noqa: E501
            )
        )
        session.commit()

        assert 2 == len(TrialIntermediateValueModel.where_trial_id(trial.trial_id, session))

        session.delete(trial)
        session.commit()

        assert 0 == len(TrialIntermediateValueModel.where_trial_id(trial.trial_id, session))


class TestTrialHeartbeatModel:
    @staticmethod
    def _create_model(session: Session) -> TrialModel:
        direction = StudyDirectionModel(direction=StudyDirection.MINIMIZE, objective=0)
        study = StudyModel(study_id=1, study_name="test-study", directions=[direction])
        trial = TrialModel(trial_id=1, study_id=study.study_id, state=TrialState.COMPLETE)
        session.add(study)
        session.add(trial)
        session.add(TrialHeartbeatModel(trial_id=trial.trial_id))
        session.commit()
        return trial

    @staticmethod
    def test_where_trial_id(session: Session) -> None:
        trial = TestTrialHeartbeatModel._create_model(session)
        trial_heartbeat = TrialHeartbeatModel.where_trial_id(trial.trial_id, session)
        assert trial_heartbeat is not None
        assert isinstance(trial_heartbeat.heartbeat, datetime)

    @staticmethod
    def test_cascade_delete_on_trial(session: Session) -> None:
        trial = TestTrialHeartbeatModel._create_model(session)
        session.commit()

        assert TrialHeartbeatModel.where_trial_id(trial.trial_id, session) is not None

        session.delete(trial)
        session.commit()

        assert TrialHeartbeatModel.where_trial_id(trial.trial_id, session) is None


class TestVersionInfoModel:
    @staticmethod
    def test_version_info_id_constraint(session: Session) -> None:
        session.add(VersionInfoModel(schema_version=1, library_version="0.0.1"))
        session.commit()

        # Test check constraint of version_info_id.
        session.add(VersionInfoModel(version_info_id=2, schema_version=2, library_version="0.0.2"))
        pytest.raises(IntegrityError, lambda: session.commit())
