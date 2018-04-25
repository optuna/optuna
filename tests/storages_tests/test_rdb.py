from datetime import datetime
from mock import Mock
from mock import patch
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from typing import Dict  # NOQA
import unittest
import uuid

from pfnopt.distributions import BaseDistribution  # NOQA
from pfnopt.distributions import CategoricalDistribution
from pfnopt.distributions import json_to_distribution
from pfnopt.distributions import UniformDistribution
from pfnopt.storages.rdb import BaseModel
from pfnopt.storages.rdb import RDBStorage
from pfnopt.storages.rdb import SCHEMA_VERSION
from pfnopt.storages.rdb import StudyModel
from pfnopt.storages.rdb import TrialModel
from pfnopt.storages.rdb import TrialParamDistributionModel
from pfnopt.storages.rdb import VersionInfoModel
import pfnopt.trial as trial_module
from pfnopt import version


def test_trial_model():
    # type: () -> None

    engine = create_engine('sqlite:///:memory:')
    session = Session(bind=engine)
    BaseModel.metadata.create_all(engine)

    datetime_1 = datetime.now()

    session.add(TrialModel())
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


class TestRDBStorage(unittest.TestCase):

    def test_init(self):
        # type: () -> None

        storage = RDBStorage('sqlite:///:memory:')
        session = storage.scoped_session()

        version_info = session.query(VersionInfoModel).first()
        assert version_info.schema_version == SCHEMA_VERSION
        assert version_info.library_version == version.__version__

    def test_create_new_study_id(self):
        # type: () -> None

        storage = self.create_test_storage()
        session = storage.scoped_session()

        study_id = storage.create_new_study_id()

        result = session.query(StudyModel).all()
        assert len(result) == 1
        assert result[0].study_id == study_id

    def test_create_new_study_id_duplicated_uuid(self):
        # type: () -> None

        mock = Mock()
        mock.side_effect = ['uuid1', 'uuid1', 'uuid2', 'uuid3']

        with patch.object(uuid, 'uuid4', mock) as mock_object:
            storage = self.create_test_storage()
            session = storage.scoped_session()

            storage.create_new_study_id()
            study_id = storage.create_new_study_id()

            result = session.query(StudyModel).filter(StudyModel.study_id == study_id).one()
            assert result.study_uuid == 'uuid2'
            assert mock_object.call_count == 3

    def test_get_study_id_from_uuid(self):
        # type: () -> None

        storage = self.create_test_storage()
        session = storage.scoped_session()

        # test not existing study
        self.assertRaises(ValueError, lambda: storage.get_study_id_from_uuid('dummy-uuid'))

        # test existing study
        storage.create_new_study_id()
        study = session.query(StudyModel).one()
        assert storage.get_study_id_from_uuid(study.study_uuid) == study.study_id

    def test_get_study_uuid_from_id(self):
        # type: () -> None

        storage = self.create_test_storage()
        session = storage.scoped_session()

        # test not existing study
        self.assertRaises(ValueError, lambda: storage.get_study_uuid_from_id(0))

        # test existing study
        storage.create_new_study_id()
        study = session.query(StudyModel).one()
        assert storage.get_study_uuid_from_id(study.study_id) == study.study_uuid

    def test_create_new_trial_id(self):
        # type: () -> None

        storage = self.create_test_storage()
        session = storage.scoped_session()

        study_id = storage.create_new_study_id()
        trial_id = storage.create_new_trial_id(study_id)

        result = session.query(TrialModel).all()
        assert len(result) == 1
        assert result[0].study_id == study_id
        assert result[0].trial_id == trial_id
        assert result[0].state == trial_module.State.RUNNING

    def test_set_trial_param_distribution(self):
        # type: () -> None

        example_distributions = {
            'x': UniformDistribution(low=1., high=2.),
            'y': CategoricalDistribution(choices=('Otemachi', 'Tokyo', 'Ginza'))
        }  # type: Dict[str, BaseDistribution]

        storage = self.create_test_storage()
        session = storage.scoped_session()
        study_id = storage.create_new_study_id()

        trial_id = storage.create_new_trial_id(study_id)
        storage.set_trial_param_distribution(trial_id, 'x', example_distributions['x'])
        storage.set_trial_param_distribution(trial_id, 'y', example_distributions['y'])

        # test setting new name
        result_1 = session.query(TrialParamDistributionModel). \
            filter(TrialParamDistributionModel.param_name == 'x').one()
        assert result_1.trial_id == trial_id
        assert json_to_distribution(result_1.distribution_json) == example_distributions['x']

        result_2 = session.query(TrialParamDistributionModel). \
            filter(TrialParamDistributionModel.param_name == 'y').one()
        assert result_2.trial_id == trial_id
        assert json_to_distribution(result_2.distribution_json) == example_distributions['y']

        # test setting a duplicated pair of trial and parameter name
        self.assertRaises(
            IntegrityError,
            lambda: storage.set_trial_param_distribution(
                trial_id,  # existing trial_id
                'x',
                example_distributions['x']))

    def test_check_table_schema_compatibility(self):
        # type: () -> None

        storage = self.create_test_storage()
        session = storage.scoped_session()

        # test not raising error for out of date schema type
        storage._check_table_schema_compatibility()

        # test raising error for out of date schema type
        version_info = session.query(VersionInfoModel).one()
        version_info.schema_version = SCHEMA_VERSION - 1
        session.commit()

        with pytest.raises(RuntimeError):
            storage._check_table_schema_compatibility()

    @staticmethod
    def create_test_storage():
        # type: () -> RDBStorage

        storage = RDBStorage('sqlite:///:memory:')
        return storage
