from datetime import datetime
import json
from typing import List  # NOQA
import unittest

from pfnopt.distributions import CategoricalDistribution
from pfnopt.distributions import json_to_distribution
from pfnopt.distributions import UniformDistribution
from pfnopt.storage.rdb import Base
from pfnopt.storage.rdb import RDBStorage
from pfnopt.storage.rdb import Study
from pfnopt.storage.rdb import StudyParam
from pfnopt.storage.rdb import Trial
from pfnopt.storage.rdb import TrialParam
from pfnopt.storage.rdb import TrialValue
import pfnopt.trial as trial_module


class TestRDBStorage(unittest.TestCase):

    def test_create_new_study_id(self):
        # type: () -> None
        storage = self.create_test_storage()

        study_id = storage.create_new_study_id()

        result = storage.session.query(Study).all()
        assert len(result) == 1
        assert result[0].study_id == study_id

        storage.close()

    def test_set_study_param_distribution(self):
        # type: () -> None
        storage = self.create_test_storage()

        uniform = UniformDistribution(low=1., high=2.)
        categorical = CategoricalDistribution(choices=('Otemachi', 'Tokyo', 'Ginza'))

        study_id = storage.create_new_study_id()
        storage.set_study_param_distribution(study_id, 'x', uniform)
        storage.set_study_param_distribution(study_id, 'y', categorical)

        # test setting new name
        result_1 = storage.session.query(StudyParam).filter(StudyParam.param_name == 'x').one()
        distribution_1 = json_to_distribution(result_1.distribution_json)
        assert isinstance(distribution_1, UniformDistribution)
        assert distribution_1.low == 1.
        assert distribution_1.high == 2.

        result_2 = storage.session.query(StudyParam).filter(StudyParam.param_name == 'y').one()
        distribution_2 = json_to_distribution(result_2.distribution_json)
        assert isinstance(distribution_2, CategoricalDistribution)
        assert distribution_2.choices == ('Otemachi', 'Tokyo', 'Ginza')

        # test setting existing name with different distribution
        self.assertRaises(
            AssertionError,
            lambda: storage.set_study_param_distribution(study_id, 'x', categorical))

        # test setting existing name with the same distribution
        storage.set_study_param_distribution(study_id, 'y', categorical)

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

        uniform = UniformDistribution(low=1., high=2.)
        categorical = CategoricalDistribution(choices=('Otemachi', 'Tokyo', 'Ginza'))
        storage.set_study_param_distribution(study_id, 'x', uniform)
        storage.set_study_param_distribution(study_id, 'y', categorical)

        def find_trial_param(items, param_name):
            # type: (List[TrialParam], str) -> TrialParam
            return next(filter(lambda p: p.study_param.param_name == param_name, items), None)

        # test setting new name
        storage.set_trial_param(trial_id, 'x', 0.5)
        storage.set_trial_param(trial_id, 'y', categorical.to_internal_repr('Ginza'))

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
            return next(filter(lambda p: p.step == step, items), None)

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
        trial_id = storage.create_new_trial_id(study_id)

        datetime_start = datetime.strptime('20180226', '%Y%m%d')
        system_attrs = trial_module.SystemAttributes(
            datetime_start=datetime_start,
            datetime_complete=None)

        uniform = UniformDistribution(low=1., high=2.)
        categorical = CategoricalDistribution(choices=('Otemachi', 'Tokyo', 'Ginza'))
        storage.set_study_param_distribution(study_id, 'x', uniform)
        storage.set_study_param_distribution(study_id, 'y', categorical)

        storage.set_trial_value(trial_id, 1.0)
        storage.set_trial_state(trial_id, trial_module.State.COMPLETE)
        storage.set_trial_param(trial_id, 'x', 0.5)
        storage.set_trial_param(trial_id, 'y', 2)
        storage.set_trial_intermediate_value(trial_id, 0, 2.0)
        storage.set_trial_intermediate_value(trial_id, 1, 3.0)
        storage.set_trial_system_attrs(trial_id, system_attrs)

        result = storage.get_trial(trial_id)
        assert result.value == 1.0
        assert result.state == trial_module.State.COMPLETE
        assert result.params['x'] == 0.5
        assert result.params['y'] == 'Ginza'
        assert result.intermediate_values[0] == 2.0
        assert result.intermediate_values[1] == 3.0
        assert result.system_attrs.datetime_start == datetime_start
        assert result.system_attrs.datetime_complete is None
        assert result.params_in_internal_repr['x'] == 0.5
        assert result.params_in_internal_repr['y'] == 2.0

        storage.close()

    def test_get_all_trials(self):
        # type: () -> None
        storage = self.create_test_storage()

        study_id_1 = storage.create_new_study_id()
        study_id_2 = storage.create_new_study_id()
        trial_id_1_1 = storage.create_new_trial_id(study_id_1)
        trial_id_1_2 = storage.create_new_trial_id(study_id_1)
        trial_id_2_1 = storage.create_new_trial_id(study_id_2)

        result_1 = storage.get_all_trials(study_id_1)
        result_trial_ids_1 = set(map(lambda x: x.trial_id, result_1))
        assert result_trial_ids_1 == {trial_id_1_1, trial_id_1_2}

        result_2 = storage.get_all_trials(study_id_2)
        result_trial_ids_2 = set(map(lambda x: x.trial_id, result_2))
        assert result_trial_ids_2 == {trial_id_2_1}

    @staticmethod
    def create_test_storage():
        # type: () -> RDBStorage
        storage = RDBStorage('sqlite:///:memory:')
        Base.metadata.create_all(storage.engine)
        return storage
