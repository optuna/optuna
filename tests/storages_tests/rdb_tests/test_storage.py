from mock import patch
import pytest
import sys
import tempfile
from typing import Dict  # NOQA

from optuna.distributions import BaseDistribution  # NOQA
from optuna.distributions import CategoricalDistribution
from optuna.distributions import json_to_distribution
from optuna.distributions import UniformDistribution
from optuna.storages.rdb.models import SCHEMA_VERSION
from optuna.storages.rdb.models import StudyModel
from optuna.storages.rdb.models import TrialParamModel
from optuna.storages.rdb.models import VersionInfoModel
from optuna.storages import RDBStorage
from optuna.structs import DuplicatedStudyError
from optuna.structs import StorageInternalError
from optuna.structs import StudyDirection
from optuna.structs import StudySummary
from optuna.structs import TrialState
from optuna import version


def test_init():
    # type: () -> None

    storage = create_test_storage()
    session = storage.scoped_session()

    version_info = session.query(VersionInfoModel).first()
    assert version_info.schema_version == SCHEMA_VERSION
    assert version_info.library_version == version.__version__


def test_init_url_template():
    # type: ()-> None

    with tempfile.NamedTemporaryFile(suffix='{SCHEMA_VERSION}') as tf:
        storage = RDBStorage('sqlite:///' + tf.name)
        assert storage.engine.url.database.endswith(str(SCHEMA_VERSION))


def test_init_db_module_import_error():
    # type: () -> None

    expected_msg = 'Failed to import DB access module for the specified storage URL. ' \
                   'Please install appropriate one.'

    with patch.dict(sys.modules, {'psycopg2': None}):
        with pytest.raises(ImportError, match=expected_msg):
            RDBStorage('postgresql://user:password@host/database')


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


def test_create_new_study_id_duplicated_name():
    # type: () -> None

    storage = create_test_storage()
    study_name = 'sample_study_name'
    storage.create_new_study_id(study_name)
    with pytest.raises(DuplicatedStudyError):
        storage.create_new_study_id(study_name)


def test_set_trial_param_to_check_distribution_json():
    # type: () -> None

    example_distributions = {
        'x': UniformDistribution(low=1., high=2.),
        'y': CategoricalDistribution(choices=('Otemachi', 'Tokyo', 'Ginza'))
    }  # type: Dict[str, BaseDistribution]

    storage = create_test_storage()
    session = storage.scoped_session()
    study_id = storage.create_new_study_id()

    trial_id = storage.create_new_trial_id(study_id)
    storage.set_trial_param(trial_id, 'x', 1.5, example_distributions['x'])
    storage.set_trial_param(trial_id, 'y', 2, example_distributions['y'])

    # test setting new name
    result_1 = session.query(TrialParamModel). \
        filter(TrialParamModel.param_name == 'x').one()
    assert json_to_distribution(result_1.distribution_json) == example_distributions['x']

    result_2 = session.query(TrialParamModel). \
        filter(TrialParamModel.param_name == 'y').one()
    assert json_to_distribution(result_2.distribution_json) == example_distributions['y']


def test_get_all_study_summaries_with_multiple_studies():
    # type: () -> None

    storage = create_test_storage()

    # Set up a MINIMIZE study.
    study_id_1 = storage.create_new_study_id()
    storage.set_study_direction(study_id_1, StudyDirection.MINIMIZE)

    trial_id_1_1 = storage.create_new_trial_id(study_id_1)
    trial_id_1_2 = storage.create_new_trial_id(study_id_1)

    storage.set_trial_value(trial_id_1_1, 100)
    storage.set_trial_value(trial_id_1_2, 0)

    storage.set_trial_state(trial_id_1_1, TrialState.COMPLETE)
    storage.set_trial_state(trial_id_1_2, TrialState.COMPLETE)

    # Set up a MAXIMIZE study.
    study_id_2 = storage.create_new_study_id()
    storage.set_study_direction(study_id_2, StudyDirection.MAXIMIZE)

    # TODO(sano): Add more trials after implementing maximize.
    trial_id_2_1 = storage.create_new_trial_id(study_id_2)

    storage.set_trial_value(trial_id_2_1, -100)

    storage.set_trial_state(trial_id_2_1, TrialState.COMPLETE)

    # Set up an empty study.
    study_id_3 = storage.create_new_study_id()

    summaries = storage.get_all_study_summaries()
    summaries = sorted(summaries, key=lambda x: x.study_id)

    expected_summary_1 = StudySummary(
        study_id=study_id_1,
        study_name=storage.get_study_name_from_id(study_id_1),
        direction=StudyDirection.MINIMIZE,
        user_attrs={},
        system_attrs={},
        best_trial=summaries[0].best_trial,  # This always passes.
        n_trials=2,
        datetime_start=summaries[0].datetime_start  # This always passes.
    )
    expected_summary_2 = StudySummary(
        study_id=study_id_2,
        study_name=storage.get_study_name_from_id(study_id_2),
        direction=StudyDirection.MAXIMIZE,
        user_attrs={},
        system_attrs={},
        best_trial=summaries[1].best_trial,  # This always passes.
        n_trials=1,
        datetime_start=summaries[1].datetime_start  # This always passes.
    )
    expected_summary_3 = StudySummary(
        study_id=study_id_3,
        study_name=storage.get_study_name_from_id(study_id_3),
        direction=StudyDirection.NOT_SET,
        user_attrs={},
        system_attrs={},
        best_trial=None,
        n_trials=0,
        datetime_start=None)

    assert summaries[0] == expected_summary_1
    assert summaries[1] == expected_summary_2
    assert summaries[2] == expected_summary_3

    assert summaries[0].best_trial is not None
    assert summaries[0].best_trial.value == 0

    assert summaries[1].best_trial is not None
    assert summaries[1].best_trial.value == -100


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


def test_commit():
    # type: () -> None

    storage = create_test_storage()
    session = storage.scoped_session()

    # This object violates the unique constraint of version_info_id.
    v = VersionInfoModel(version_info_id=1, schema_version=1, library_version='0.0.1')
    session.add(v)
    with pytest.raises(StorageInternalError):
        storage._commit(session)
