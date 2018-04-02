from datetime import datetime
import json
from mock import Mock
from mock import patch
import pytest
from sqlalchemy.exc import IntegrityError
from typing import Dict  # NOQA
from typing import List  # NOQA
import unittest
import uuid

from pfnopt.distributions import BaseDistribution  # NOQA
from pfnopt.distributions import CategoricalDistribution
from pfnopt.distributions import json_to_distribution
from pfnopt.distributions import UniformDistribution
from pfnopt.storages.rdb import RDBStorage
from pfnopt.storages.rdb import SCHEMA_VERSION
from pfnopt.storages.rdb import Study
from pfnopt.storages.rdb import Trial
from pfnopt.storages.rdb import TrialParam
from pfnopt.storages.rdb import TrialParamDistribution
from pfnopt.storages.rdb import TrialValue
from pfnopt.storages.rdb import VersionInfo
import pfnopt.trial as trial_module
from pfnopt import version


class TestRDBStorage(unittest.TestCase):

    def test_init(self):
        storage = RDBStorage('sqlite:///:memory:')

        version_info = storage.session.query(VersionInfo).first()
        assert version_info.schema_version == SCHEMA_VERSION
        assert version_info.library_version == version.__version__

        storage.close()

    def test_create_new_study_id(self):
        # type: () -> None

        storage = self.create_test_storage()

        study_id = storage.create_new_study_id()

        result = storage.session.query(Study).all()
        assert len(result) == 1
        assert result[0].study_id == study_id

        storage.close()

    def test_create_new_study_id_duplicated_uuid(self):
        # type: () -> None

        mock = Mock()
        mock.side_effect = ['uuid1', 'uuid1', 'uuid2', 'uuid3']

        with patch.object(uuid, 'uuid4', mock) as mock_object:
            storage = self.create_test_storage()

            storage.create_new_study_id()
            study_id = storage.create_new_study_id()

            result = storage.session.query(Study).filter(Study.study_id == study_id).one()
            assert result.study_uuid == 'uuid2'
            assert mock_object.call_count == 3

            storage.close()

    def test_get_study_id_from_uuid(self):
        # type: () -> None

        storage = self.create_test_storage()

        # test not existing study
        self.assertRaises(ValueError, lambda: storage.get_study_id_from_uuid('dummy-uuid'))

        # test existing study
        storage.create_new_study_id()
        study = storage.session.query(Study).one()
        assert storage.get_study_id_from_uuid(study.study_uuid) == study.study_id

    def test_get_study_uuid_from_id(self):
        # type: () -> None

        storage = self.create_test_storage()

        # test not existing study
        self.assertRaises(ValueError, lambda: storage.get_study_uuid_from_id(0))

        # test existing study
        storage.create_new_study_id()
        study = storage.session.query(Study).one()
        assert storage.get_study_uuid_from_id(study.study_id) == study.study_uuid

    def test_set_trial_param_distribution(self):
        # type: () -> None

        storage = self.create_test_storage()
        study_id = storage.create_new_study_id()

        trial_id = storage.create_new_trial_id(study_id)
        storage.set_trial_param_distribution(trial_id, 'x', self.example_distributions['x'])
        storage.set_trial_param_distribution(trial_id, 'y', self.example_distributions['y'])

        # test setting new name
        result_1 = storage.session.query(TrialParamDistribution). \
            filter(TrialParamDistribution.param_name == 'x').one()
        distribution_1 = json_to_distribution(result_1.distribution_json)
        assert result_1.trial_id == trial_id
        assert isinstance(distribution_1, UniformDistribution)
        assert distribution_1.low == self.example_distributions['x'].low
        assert distribution_1.high == self.example_distributions['x'].high

        result_2 = storage.session.query(TrialParamDistribution). \
            filter(TrialParamDistribution.param_name == 'y').one()
        distribution_2 = json_to_distribution(result_2.distribution_json)
        assert result_2.trial_id == trial_id
        assert isinstance(distribution_2, CategoricalDistribution)
        assert distribution_2.choices == self.example_distributions['y'].choices

        # test setting existing name with the same distribution
        storage.set_trial_param_distribution(
            storage.create_new_trial_id(study_id),  # new trial_id
            'y',
            self.example_distributions['y'])

        # test setting existing name with different distribution kind
        self.assertRaises(
            ValueError,
            lambda: storage.set_trial_param_distribution(
                storage.create_new_trial_id(study_id),  # new trial_id
                'x',
                self.example_distributions['y']))

        # test setting existing name with different value (CategoricalDistribution)
        self.assertRaises(
            ValueError,
            lambda: storage.set_trial_param_distribution(
                storage.create_new_trial_id(study_id),  # new trial_id
                'y',
                self.example_distributions['y']._replace(choices=('Tokyo', 'Shinbashi'))))

        # test setting existing name with different value (non CategoricalDistribution)
        storage.set_trial_param_distribution(
            storage.create_new_trial_id(study_id),  # new trial_id
            'x',
            self.example_distributions['x']._replace(low=100, high=200))

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

        study_id = storage.create_new_study_id()
        trial_id = storage.create_new_trial_id(study_id)

        result = storage.session.query(Trial).all()
        assert len(result) == 1
        assert result[0].study_id == study_id
        assert result[0].trial_id == trial_id
        assert result[0].state == trial_module.State.RUNNING

        storage.close()

    def test_set_trial_state(self):
        # type: () -> None

        storage = self.create_test_storage()

        study_id = storage.create_new_study_id()
        trial_id = storage.create_new_trial_id(study_id)

        result_1 = storage.session.query(Trial).filter(Trial.trial_id == trial_id).one().state

        storage.set_trial_state(trial_id, trial_module.State.PRUNED)

        result_2 = storage.session.query(Trial).filter(Trial.trial_id == trial_id).one().state

        assert result_1 == trial_module.State.RUNNING
        assert result_2 == trial_module.State.PRUNED

        storage.close()

    def test_set_trial_param(self):
        # type: () -> None

        storage = self.create_test_storage()

        study_id = storage.create_new_study_id()
        trial_id = storage.create_new_trial_id(study_id)

        self.set_distributions(storage, study_id, self.example_distributions)

        def find_trial_param(items, param_name):
            # type: (List[TrialParam], str) -> TrialParam
            return [p for p in items if p.param_distribution.param_name == param_name][0]

        # test setting new name
        storage.set_trial_param(trial_id, 'x', 0.5)
        storage.set_trial_param(trial_id, 'y', 2.)

        result = storage.session.query(TrialParam).filter(TrialParam.trial_id == trial_id).all()
        assert len(result) == 2
        assert find_trial_param(result, 'x').param_value == 0.5
        assert find_trial_param(result, 'y').param_value == 2.

        # test setting existing name with different value
        self.assertRaises(AssertionError, lambda: storage.set_trial_param(trial_id, 'x', 1.0))

        # test setting existing name with the same value
        storage.set_trial_param(trial_id, 'x', 0.5)

        storage.close()

    def test_set_trial_value(self):
        # type: () -> None

        storage = self.create_test_storage()

        study_id = storage.create_new_study_id()
        trial_id = storage.create_new_trial_id(study_id)

        # test setting new value
        storage.set_trial_value(trial_id, 0.5)

        result_1 = storage.session.query(Trial).filter(Trial.trial_id == trial_id).one()
        assert result_1.value == 0.5

        storage.close()

    def test_set_trial_intermediate_value(self):
        # type: () -> None

        storage = self.create_test_storage()

        study_id = storage.create_new_study_id()
        trial_id = storage.create_new_trial_id(study_id)

        def find_trial_value(items, step):
            # type: (List[TrialValue], int) -> TrialValue
            return [p for p in items if p.step == step][0]

        # test setting new values
        storage.set_trial_intermediate_value(trial_id, 0, 0.3)
        storage.set_trial_intermediate_value(trial_id, 2, 0.4)

        result_1 = storage.session.query(TrialValue).filter(TrialValue.trial_id == trial_id).all()
        assert len(result_1) == 2
        assert find_trial_value(result_1, 0).trial_id == trial_id
        assert find_trial_value(result_1, 0).value == 0.3
        assert find_trial_value(result_1, 2).trial_id == trial_id
        assert find_trial_value(result_1, 2).value == 0.4

        # test setting existing step with different value
        self.assertRaises(
            AssertionError,
            lambda: storage.set_trial_intermediate_value(trial_id, 0, 0.5))

        # test setting existing step with the same value
        storage.set_trial_intermediate_value(trial_id, 0, 0.3)

        storage.close()

    def test_set_trial_system_attrs(self):
        # type: () -> None

        storage = self.create_test_storage()

        study_id = storage.create_new_study_id()
        trial_id = storage.create_new_trial_id(study_id)

        # test setting value
        system_attrs_1 = trial_module.SystemAttributes(
            datetime_start=datetime.strptime('20180226', '%Y%m%d'),
            datetime_complete=None)
        storage.set_trial_system_attrs(trial_id, system_attrs_1)

        result_1 = storage.session.query(Trial).filter(Trial.trial_id == trial_id).one()
        system_attr_json_1 = json.loads(result_1.system_attributes_json)
        assert len(system_attr_json_1) == 2
        assert system_attr_json_1['datetime_start'] == '20180226000000'
        assert system_attr_json_1['datetime_complete'] is None

        # test overwriting value
        system_attrs_2 = system_attrs_1._replace(
            datetime_complete=datetime.strptime('20180227', '%Y%m%d'))
        storage.set_trial_system_attrs(trial_id, system_attrs_2)

        result_2 = storage.session.query(Trial).filter(Trial.trial_id == trial_id).one()
        system_attr_json_2 = json.loads(result_2.system_attributes_json)
        assert len(system_attr_json_1) == 2
        assert system_attr_json_2['datetime_start'] == '20180226000000'
        assert system_attr_json_2['datetime_complete'] == '20180227000000'

        storage.close()

    def test_get_trial(self):
        # type: () -> None

        storage = self.create_test_storage()
        study_id = storage.create_new_study_id()

        trial_id = TestRDBStorage.create_new_trial_with_example_trial(
            storage, study_id, self.example_distributions, self.example_trials[0])

        result = storage.get_trial(trial_id)
        assert result == TestRDBStorage.example_trials[0]._replace(trial_id=trial_id)

        storage.close()

    def test_get_all_trials(self):
        # type: () -> None
        storage = self.create_test_storage()
        study_id_1 = storage.create_new_study_id()
        study_id_2 = storage.create_new_study_id()

        trial_id_1_1 = self.create_new_trial_with_example_trial(
            storage, study_id_1, self.example_distributions, self.example_trials[0])
        trial_id_1_2 = self.create_new_trial_with_example_trial(
            storage, study_id_1, self.example_distributions, self.example_trials[1])
        trial_id_2_1 = self.create_new_trial_with_example_trial(
            storage, study_id_2, self.example_distributions, self.example_trials[0])

        # test getting multiple trials
        result_1 = storage.get_all_trials(study_id_1)
        assert sorted(result_1) == sorted([
            self.example_trials[0]._replace(trial_id=trial_id_1_1),
            self.example_trials[1]._replace(trial_id=trial_id_1_2)])

        # test getting trials per study
        result_2 = storage.get_all_trials(study_id_2)
        assert result_2 == [self.example_trials[0]._replace(trial_id=trial_id_2_1)]

    example_distributions = {
        'x': UniformDistribution(low=1., high=2.),
        'y': CategoricalDistribution(choices=('Otemachi', 'Tokyo', 'Ginza'))
    }  # type: Dict[str, BaseDistribution]

    example_trials = [
        trial_module.Trial(
            trial_id=-1,  # dummy id
            value=1.,
            state=trial_module.State.COMPLETE,
            system_attrs=trial_module.SystemAttributes(
                datetime_start=datetime.strptime('20180226', '%Y%m%d'),
                datetime_complete=None),
            user_attrs={},
            params={'x': 0.5, 'y': 'Ginza'},
            intermediate_values={0: 2., 1: 3.},
            params_in_internal_repr={'x': .5, 'y': 2.}
        ),
        trial_module.Trial(
            trial_id=-1,  # dummy id
            value=2.,
            state=trial_module.State.PRUNED,
            system_attrs=trial_module.SystemAttributes(
                datetime_start=datetime.strptime('20180227', '%Y%m%d'),
                datetime_complete=datetime.strptime('20180228', '%Y%m%d')),
            user_attrs={},
            params={'x': 0.01, 'y': 'Otemachi'},
            intermediate_values={0: -2., 1: -3., 2: 100.},
            params_in_internal_repr={'x': .01, 'y': 0.}
        )
    ]

    def test_check_table_schema_compatibility(self):
        storage = self.create_test_storage()

        # test not raising error for out of date schema type
        storage._check_table_schema_compatibility()

        # test raising error for out of date schema type
        version_info = storage.session.query(VersionInfo).one()
        version_info.schema_version = SCHEMA_VERSION - 1
        storage.session.commit()

        with pytest.raises(RuntimeError):
            storage._check_table_schema_compatibility()

    @staticmethod
    def create_new_trial_with_example_trial(storage, study_id, distributions, example_trial):
        # type: (RDBStorage, int, Dict[str, BaseDistribution], trial_module.Trial) -> int

        trial_id = storage.create_new_trial_id(study_id)

        storage.set_trial_value(trial_id, example_trial.value)
        storage.set_trial_state(trial_id, example_trial.state)
        storage.set_trial_system_attrs(trial_id, example_trial.system_attrs)
        TestRDBStorage.set_distributions(storage, trial_id, distributions)

        for name, ex_repr in example_trial.params.items():
            storage.set_trial_param(trial_id, name, distributions[name].to_internal_repr(ex_repr))

        for step, value in example_trial.intermediate_values.items():
            storage.set_trial_intermediate_value(trial_id, step, value)

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
