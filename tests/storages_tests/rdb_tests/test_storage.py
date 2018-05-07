from mock import Mock
from mock import patch
import pytest
from sqlalchemy.exc import IntegrityError
from typing import Dict  # NOQA
import uuid

from pfnopt.distributions import BaseDistribution  # NOQA
from pfnopt.distributions import CategoricalDistribution
from pfnopt.distributions import json_to_distribution
from pfnopt.distributions import UniformDistribution
from pfnopt.storages.base import SYSTEM_ATTRS_KEY
from pfnopt.storages.rdb.models import SCHEMA_VERSION
from pfnopt.storages.rdb.models import StudyModel
from pfnopt.storages.rdb.models import TrialParamDistributionModel
from pfnopt.storages.rdb.models import VersionInfoModel
from pfnopt.storages import RDBStorage
from pfnopt.structs import StudySummary
from pfnopt.structs import StudyTask
from pfnopt import version


def test_init():
    # type: () -> None

    storage = create_test_storage()
    session = storage.scoped_session()

    version_info = session.query(VersionInfoModel).first()
    assert version_info.schema_version == SCHEMA_VERSION
    assert version_info.library_version == version.__version__


def test_create_new_study_id_multiple_studies():
    # type: () -> None

    storage = create_test_storage()
    session = storage.scoped_session()

    study_id_1 = storage.create_new_study_id()
    study_id_2 = storage.create_new_study_id()

    result = session.query(StudyModel).all()
    result = sorted(result, key=lambda x: x.study_id)
    assert len(result) == 2
    assert result[0].study_id == study_id_1
    assert result[1].study_id == study_id_2


def test_create_new_study_id_duplicated_uuid():
    # type: () -> None

    mock = Mock()
    mock.side_effect = ['uuid1', 'uuid1', 'uuid2', 'uuid3']

    with patch.object(uuid, 'uuid4', mock) as mock_object:
        storage = create_test_storage()
        session = storage.scoped_session()

        storage.create_new_study_id()
        study_id = storage.create_new_study_id()

        result = session.query(StudyModel).filter(StudyModel.study_id == study_id).one()
        assert result.study_uuid == 'uuid2'
        assert mock_object.call_count == 3


def test_set_trial_param_distribution():
    # type: () -> None

    example_distributions = {
        'x': UniformDistribution(low=1., high=2.),
        'y': CategoricalDistribution(choices=('Otemachi', 'Tokyo', 'Ginza'))
    }  # type: Dict[str, BaseDistribution]

    storage = create_test_storage()
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
    with pytest.raises(IntegrityError):
        storage.set_trial_param_distribution(
            trial_id,  # existing trial_id
            'x',
            example_distributions['x'])


def test_get_all_study_summaries_with_multiple_studies():
    # type: () -> None

    storage = create_test_storage()

    study_id_1 = storage.create_new_study_id()
    study_id_2 = storage.create_new_study_id()

    storage.set_study_task(study_id_1, StudyTask.MINIMIZE)
    storage.set_study_task(study_id_2, StudyTask.MAXIMIZE)

    storage.create_new_trial_id(study_id_1)
    storage.create_new_trial_id(study_id_1)
    storage.create_new_trial_id(study_id_2)

    summaries = storage.get_all_study_summaries()
    summaries = sorted(summaries, key=lambda x: x.study_id)

    expected_summary_1 = StudySummary(
        study_id=study_id_1,
        study_uuid=storage.get_study_uuid_from_id(study_id_1),
        task=StudyTask.MINIMIZE,
        user_attrs={SYSTEM_ATTRS_KEY: {}},
        best_trial=None,
        n_trials=2,
        datetime_start=summaries[0].datetime_start  # This always passes.
    )
    expected_summary_2 = StudySummary(
        study_id=study_id_2,
        study_uuid=storage.get_study_uuid_from_id(study_id_2),
        task=StudyTask.MAXIMIZE,
        user_attrs={SYSTEM_ATTRS_KEY: {}},
        best_trial=None,
        n_trials=1,
        datetime_start=summaries[1].datetime_start  # This always passes.
    )

    assert summaries[0] == expected_summary_1
    assert summaries[1] == expected_summary_2


def test_check_table_schema_compatibility():
    # type: () -> None

    storage = create_test_storage()
    session = storage.scoped_session()

    # test not raising error for out of date schema type
    storage._check_table_schema_compatibility()

    # test raising error for out of date schema type
    version_info = session.query(VersionInfoModel).one()
    version_info.schema_version = SCHEMA_VERSION - 1
    session.commit()

    with pytest.raises(RuntimeError):
        storage._check_table_schema_compatibility()


def create_test_storage():
    # type: () -> RDBStorage

    storage = RDBStorage('sqlite:///:memory:')
    return storage
