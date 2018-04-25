from datetime import datetime
from mock import Mock
from mock import patch
import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from typing import Dict  # NOQA
from typing import List  # NOQA
import unittest
import uuid

from pfnopt.distributions import BaseDistribution  # NOQA
from pfnopt.distributions import CategoricalDistribution
from pfnopt.distributions import json_to_distribution
from pfnopt.distributions import UniformDistribution
from pfnopt.storages.base import SYSTEM_ATTRS_KEY
from pfnopt.storages.rdb import BaseModel
from pfnopt.storages.rdb import RDBStorage
from pfnopt.storages.rdb import SCHEMA_VERSION
from pfnopt.storages.rdb import StudyModel
from pfnopt.storages.rdb import TrialModel
from pfnopt.storages.rdb import TrialParamDistributionModel
from pfnopt.storages.rdb import TrialParamModel
from pfnopt.storages.rdb import TrialValueModel
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

    def test_set_trial_param_distribution(self):
        # type: () -> None

        storage = self.create_test_storage()
        session = storage.scoped_session()
        study_id = storage.create_new_study_id()

        trial_id = storage.create_new_trial_id(study_id)
        storage.set_trial_param_distribution(trial_id, 'x', self.example_distributions['x'])
        storage.set_trial_param_distribution(trial_id, 'y', self.example_distributions['y'])

        # test setting new name
        result_1 = session.query(TrialParamDistributionModel). \
            filter(TrialParamDistributionModel.param_name == 'x').one()
        assert result_1.trial_id == trial_id
        assert json_to_distribution(result_1.distribution_json) == self.example_distributions['x']

        result_2 = session.query(TrialParamDistributionModel). \
            filter(TrialParamDistributionModel.param_name == 'y').one()
        assert result_2.trial_id == trial_id
        assert json_to_distribution(result_2.distribution_json) == self.example_distributions['y']

        # test setting a duplicated pair of trial and parameter name
        self.assertRaises(
            IntegrityError,
            lambda: storage.set_trial_param_distribution(
                trial_id,  # existing trial_id
                'x',
                self.example_distributions['x']))

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

    def test_set_trial_state(self):
        # type: () -> None

        storage = self.create_test_storage()
        session = storage.scoped_session()

        study_id = storage.create_new_study_id()
        trial_id = storage.create_new_trial_id(study_id)

        result_1 = session.query(TrialModel).filter(TrialModel.trial_id == trial_id).one().state

        storage.set_trial_state(trial_id, trial_module.State.PRUNED)

        result_2 = session.query(TrialModel).filter(TrialModel.trial_id == trial_id).one().state

        assert result_1 == trial_module.State.RUNNING
        assert result_2 == trial_module.State.PRUNED

    def test_set_trial_param(self):
        # type: () -> None

        storage = self.create_test_storage()
        session = storage.scoped_session()

        study_id = storage.create_new_study_id()
        trial_id = storage.create_new_trial_id(study_id)

        self.set_distributions(storage, study_id, self.example_distributions)

        def find_trial_param(items, param_name):
            # type: (List[TrialParamModel], str) -> TrialParamModel
            return [p for p in items if p.param_distribution.param_name == param_name][0]

        # test setting new name
        storage.set_trial_param(trial_id, 'x', 0.5)
        storage.set_trial_param(trial_id, 'y', 2.)

        result = session.query(TrialParamModel).filter(TrialParamModel.trial_id == trial_id).all()
        assert len(result) == 2
        assert find_trial_param(result, 'x').param_value == 0.5
        assert find_trial_param(result, 'y').param_value == 2.

        # test setting existing name with different value
        self.assertRaises(AssertionError, lambda: storage.set_trial_param(trial_id, 'x', 1.0))

        # test setting existing name with the same value
        storage.set_trial_param(trial_id, 'x', 0.5)

    example_distributions = {
        'x': UniformDistribution(low=1., high=2.),
        'y': CategoricalDistribution(choices=('Otemachi', 'Tokyo', 'Ginza'))
    }  # type: Dict[str, BaseDistribution]

    example_trials = [
        trial_module.Trial(
            trial_id=-1,  # dummy id
            value=1.,
            state=trial_module.State.COMPLETE,
            user_attrs={SYSTEM_ATTRS_KEY: {}},
            params={'x': 0.5, 'y': 'Ginza'},
            intermediate_values={0: 2., 1: 3.},
            params_in_internal_repr={'x': .5, 'y': 2.},
            datetime_start=None,  # dummy
            datetime_complete=None  # dummy
        ),
        trial_module.Trial(
            trial_id=-1,  # dummy id
            value=2.,
            state=trial_module.State.RUNNING,
            user_attrs={
                SYSTEM_ATTRS_KEY: {'some_key': 'some_value'},
                'tags': ['video', 'classification'], 'dataset': 'YouTube-8M'},
            params={'x': 0.01, 'y': 'Otemachi'},
            intermediate_values={0: -2., 1: -3., 2: 100.},
            params_in_internal_repr={'x': .01, 'y': 0.},
            datetime_start=None,  # dummy
            datetime_complete=None  # dummy
        )
    ]

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
    def check_example_trial_static_attributes(trial_1, trial_2):
        # type: (trial_module.Trial, trial_module.Trial) -> None

        trial_1 = trial_1._replace(trial_id=-1, datetime_start=None, datetime_complete=None)
        trial_2 = trial_2._replace(trial_id=-1, datetime_start=None, datetime_complete=None)
        assert trial_1 == trial_2

    @staticmethod
    def create_new_trial_with_example_trial(storage, study_id, distributions, example_trial):
        # type: (RDBStorage, int, Dict[str, BaseDistribution], trial_module.Trial) -> int

        trial_id = storage.create_new_trial_id(study_id)

        storage.set_trial_value(trial_id, example_trial.value)
        storage.set_trial_state(trial_id, example_trial.state)
        TestRDBStorage.set_distributions(storage, trial_id, distributions)

        for name, ex_repr in example_trial.params.items():
            storage.set_trial_param(trial_id, name, distributions[name].to_internal_repr(ex_repr))

        for step, value in example_trial.intermediate_values.items():
            storage.set_trial_intermediate_value(trial_id, step, value)

        for key, value in example_trial.user_attrs.items():
            storage.set_trial_user_attr(trial_id, key, value)

        return trial_id

    @staticmethod
    def set_distributions(storage, trial_id, distributions):
        # type: (RDBStorage, int, Dict[str, BaseDistribution]) -> None

        for k, v in distributions.items():
            storage.set_trial_param_distribution(trial_id, k, v)

    @staticmethod
    def create_test_storage():
        # type: () -> RDBStorage

        storage = RDBStorage('sqlite:///:memory:')
        return storage
