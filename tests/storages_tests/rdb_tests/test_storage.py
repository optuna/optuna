import pickle
import sys
import tempfile

from mock import patch
import pytest

from optuna.distributions import CategoricalDistribution
from optuna.distributions import json_to_distribution
from optuna.distributions import UniformDistribution
from optuna.exceptions import DuplicatedStudyError
from optuna.exceptions import StorageInternalError
from optuna.storages.rdb.models import SCHEMA_VERSION
from optuna.storages.rdb.models import StudyModel
from optuna.storages.rdb.models import TrialModel
from optuna.storages.rdb.models import TrialParamModel
from optuna.storages.rdb.models import VersionInfoModel
from optuna.storages import RDBStorage
from optuna.structs import StudyDirection
from optuna.structs import StudySummary
from optuna.structs import TrialState
from optuna import type_checking
from optuna import version

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.structs import FrozenTrial  # NOQA


def test_init():
    # type: () -> None

    storage = create_test_storage()
    session = storage.scoped_session()

    version_info = session.query(VersionInfoModel).first()
    assert version_info.schema_version == SCHEMA_VERSION
    assert version_info.library_version == version.__version__

    assert storage.get_current_version() == storage.get_head_version()
    assert storage.get_all_versions() == ['v1.2.0.a', 'v0.9.0.a']


def test_init_url_template():
    # type: ()-> None

    with tempfile.NamedTemporaryFile(suffix='{SCHEMA_VERSION}') as tf:
        storage = RDBStorage('sqlite:///' + tf.name)
        assert storage.engine.url.database.endswith(str(SCHEMA_VERSION))


def test_init_url_that_contains_percent_character():
    # type: ()-> None

    # Alembic's ini file regards '%' as the special character for variable expansion.
    # We checks `RDBStorage` does not raise an error even if a storage url contains the character.
    with tempfile.NamedTemporaryFile(suffix='%') as tf:
        RDBStorage('sqlite:///' + tf.name)

    with tempfile.NamedTemporaryFile(suffix='%foo') as tf:
        RDBStorage('sqlite:///' + tf.name)

    with tempfile.NamedTemporaryFile(suffix='%foo%%bar') as tf:
        RDBStorage('sqlite:///' + tf.name)


def test_init_db_module_import_error():
    # type: () -> None

    expected_msg = 'Failed to import DB access module for the specified storage URL. ' \
                   'Please install appropriate one.'

    with patch.dict(sys.modules, {'psycopg2': None}):
        with pytest.raises(ImportError, match=expected_msg):
            RDBStorage('postgresql://user:password@host/database')


def test_engine_kwargs():
    # type: () -> None

    create_test_storage(engine_kwargs={'pool_size': 5})

    with pytest.raises(TypeError):
        create_test_storage(engine_kwargs={'wrong_key': 'wrong_value'})


@pytest.mark.parametrize('url,engine_kwargs,expected', [
    ('mysql://localhost', {'pool_pre_ping': False}, False),
    ('mysql://localhost', {'pool_pre_ping': True}, True),
    ('mysql://localhost', {}, True),
    ('mysql+pymysql://localhost', {}, True),
    ('mysql://localhost', {'pool_size': 5}, True),
])
def test_set_default_engine_kwargs_for_mysql(url, engine_kwargs, expected):
    # type: (str, Dict[str, Any], bool)-> None

    RDBStorage._set_default_engine_kwargs_for_mysql(url, engine_kwargs)
    assert engine_kwargs['pool_pre_ping'] is expected


def test_set_default_engine_kwargs_for_mysql_with_other_rdb():
    # type: ()-> None

    # Do not change engine_kwargs if database is not MySQL.
    engine_kwargs = {}  # type: Dict[str, Any]
    RDBStorage._set_default_engine_kwargs_for_mysql('sqlite:///example.db', engine_kwargs)
    assert 'pool_pre_ping' not in engine_kwargs
    RDBStorage._set_default_engine_kwargs_for_mysql('postgres:///example.db', engine_kwargs)
    assert 'pool_pre_ping' not in engine_kwargs


def test_create_new_study_multiple_studies():
    # type: () -> None

    storage = create_test_storage()
    session = storage.scoped_session()

    study_id_1 = storage.create_new_study()
    study_id_2 = storage.create_new_study()

    result = session.query(StudyModel).all()
    result = sorted(result, key=lambda x: x.study_id)
    assert len(result) == 2
    assert result[0].study_id == study_id_1
    assert result[1].study_id == study_id_2


def test_create_new_study_duplicated_name():
    # type: () -> None

    storage = create_test_storage()
    study_name = 'sample_study_name'
    storage.create_new_study(study_name)
    with pytest.raises(DuplicatedStudyError):
        storage.create_new_study(study_name)


def test_set_trial_param_to_check_distribution_json():
    # type: () -> None

    example_distributions = {
        'x': UniformDistribution(low=1., high=2.),
        'y': CategoricalDistribution(choices=('Otemachi', 'Tokyo', 'Ginza'))
    }  # type: Dict[str, BaseDistribution]

    storage = create_test_storage()
    session = storage.scoped_session()
    study_id = storage.create_new_study()

    trial_id = storage.create_new_trial(study_id)
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
    study_id_1 = storage.create_new_study()
    storage.set_study_direction(study_id_1, StudyDirection.MINIMIZE)

    trial_id_1_1 = storage.create_new_trial(study_id_1)
    trial_id_1_2 = storage.create_new_trial(study_id_1)

    storage.set_trial_value(trial_id_1_1, 100)
    storage.set_trial_value(trial_id_1_2, 0)

    storage.set_trial_state(trial_id_1_1, TrialState.COMPLETE)
    storage.set_trial_state(trial_id_1_2, TrialState.COMPLETE)

    # Set up a MAXIMIZE study.
    study_id_2 = storage.create_new_study()
    storage.set_study_direction(study_id_2, StudyDirection.MAXIMIZE)

    trial_id_2_1 = storage.create_new_trial(study_id_2)
    trial_id_2_2 = storage.create_new_trial(study_id_2)

    storage.set_trial_value(trial_id_2_1, -100)
    storage.set_trial_value(trial_id_2_2, -200)

    storage.set_trial_state(trial_id_2_1, TrialState.COMPLETE)
    storage.set_trial_state(trial_id_2_2, TrialState.COMPLETE)

    # Set up an empty study.
    study_id_3 = storage.create_new_study()

    summaries = storage.get_all_study_summaries()
    summaries = sorted(summaries)

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
        n_trials=2,
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

    # The schema version of a newly created storage is always up-to-date.
    storage._version_manager.check_table_schema_compatibility()

    # `SCHEMA_VERSION` has not been used for compatibility check since alembic was introduced.
    version_info = session.query(VersionInfoModel).one()
    version_info.schema_version = SCHEMA_VERSION - 1
    session.commit()

    storage._version_manager.check_table_schema_compatibility()

    # TODO(ohta): Remove the following comment out when the second revision is introduced.
    # with pytest.raises(RuntimeError):
    #     storage._set_alembic_revision(storage._version_manager._get_base_version())
    #     storage._check_table_schema_compatibility()


def create_test_storage(engine_kwargs=None):
    # type: (Optional[Dict[str, Any]]) -> RDBStorage

    storage = RDBStorage('sqlite:///:memory:', engine_kwargs=engine_kwargs)
    return storage


def test_pickle_storage():
    # type: () -> None

    storage = create_test_storage()
    restored_storage = pickle.loads(pickle.dumps(storage))
    assert storage.url == restored_storage.url
    assert storage.engine_kwargs == restored_storage.engine_kwargs
    assert storage.skip_compatibility_check == restored_storage.skip_compatibility_check
    assert storage.engine != restored_storage.engine
    assert storage.scoped_session != restored_storage.scoped_session
    assert storage._version_manager != restored_storage._version_manager
    assert storage._finished_trials_cache != restored_storage._finished_trials_cache


def test_commit():
    # type: () -> None

    storage = create_test_storage()
    session = storage.scoped_session()

    # This object violates the unique constraint of version_info_id.
    v = VersionInfoModel(version_info_id=1, schema_version=1, library_version='0.0.1')
    session.add(v)
    with pytest.raises(StorageInternalError):
        storage._commit(session)


def test_create_new_trial_number():
    # type: () -> None

    storage = create_test_storage()
    study_id = storage.create_new_study()

    trial_id = storage.create_new_trial(study_id)
    assert storage._create_new_trial_number(trial_id) == 0

    trial_id = storage.create_new_trial(study_id)
    assert storage._create_new_trial_number(trial_id) == 1


def test_update_finished_trial():
    # type: () -> None

    storage = create_test_storage()
    study_id = storage.create_new_study()

    # Running trials are allowed to be updated.
    trial_id = storage.create_new_trial(study_id)
    assert storage.get_trial(trial_id).state == TrialState.RUNNING

    storage.set_trial_intermediate_value(trial_id, 3, 5)
    storage.set_trial_value(trial_id, 10)
    storage.set_trial_param(trial_id, 'x', 1.5, UniformDistribution(low=1.0, high=2.0))
    storage.set_trial_user_attr(trial_id, 'foo', 'bar')
    storage.set_trial_system_attr(trial_id, 'baz', 'qux')
    storage.set_trial_state(trial_id, TrialState.COMPLETE)

    # Finished trials are not allowed to be updated.
    for state in [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL]:
        trial_id = storage.create_new_trial(study_id)
        storage.set_trial_state(trial_id, state)

        with pytest.raises(RuntimeError):
            storage.set_trial_intermediate_value(trial_id, 3, 5)
        with pytest.raises(RuntimeError):
            storage.set_trial_value(trial_id, 10)
        with pytest.raises(RuntimeError):
            storage.set_trial_param(trial_id, 'x', 1.5, UniformDistribution(low=1.0, high=2.0))
        with pytest.raises(RuntimeError):
            storage.set_trial_user_attr(trial_id, 'foo', 'bar')
        with pytest.raises(RuntimeError):
            storage.set_trial_system_attr(trial_id, 'baz', 'qux')
        with pytest.raises(RuntimeError):
            storage.set_trial_state(trial_id, TrialState.COMPLETE)


def test_upgrade():
    # type: () -> None

    storage = create_test_storage()

    # `upgrade()` has no effect because the storage version is already up-to-date.
    old_version = storage.get_current_version()
    storage.upgrade()
    new_version = storage.get_current_version()

    assert old_version == new_version


def test_storage_cache():
    # type: () -> None

    def setup_trials(storage, study_id):
        # type: (RDBStorage, int) -> List[FrozenTrial]

        for state in [TrialState.RUNNING, TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL]:
            trial_id = storage.create_new_trial(study_id)
            storage.set_trial_state(trial_id, state)

        trials = storage.get_all_trials(study_id)
        assert len(trials) == 4

        return trials

    storage = create_test_storage()
    study_id = storage.create_new_study()
    trials = setup_trials(storage, study_id)

    with patch.object(
            TrialModel, 'find_or_raise_by_id',
            wraps=TrialModel.find_or_raise_by_id) as mock_object:
        for trial in trials:
            assert storage.get_trial(trial._trial_id) == trial
        assert mock_object.call_count == 1  # Only a running trial was fetched from the storage.

    # Running trials are fetched from the storage individually.
    with patch.object(TrialModel, 'where_study', wraps=TrialModel.where_study) as mock_object:
        assert storage.get_all_trials(study_id) == trials
        assert mock_object.call_count == 0  # `TrialModel.where_study` has not been called.

    with patch.object(
            TrialModel, 'find_or_raise_by_id',
            wraps=TrialModel.find_or_raise_by_id) as mock_object:
        assert storage.get_all_trials(study_id) == trials
        assert mock_object.call_count == 1


def test_check_python_version():
    # type: () -> None

    error_versions = [{"major": 3, "minor": 4, "micro": i} for i in range(0, 4)]
    valid_versions = [
        {"major": 2, "minor": 7, "micro": 3},
        {"major": 3, "minor": 3, "micro": 7},
        {"major": 3, "minor": 4, "micro": 4},
        {"major": 3, "minor": 4, "micro": 10},
        {"major": 3, "minor": 7, "micro": 4},
    ]

    with patch.object(sys, 'version_info') as v_info:
        # If Python version is 3.4.0 to 3.4.3, RDBStorage raises RuntimeError.
        for ver in error_versions:
            v_info.major = ver["major"]
            v_info.minor = ver["minor"]
            v_info.micro = ver["micro"]

            with pytest.raises(RuntimeError):
                RDBStorage._check_python_version()

        # Otherwise, RDBStorage does not raise RuntimeError.
        for ver in valid_versions:
            v_info.major = ver["major"]
            v_info.minor = ver["minor"]
            v_info.micro = ver["micro"]
            RDBStorage._check_python_version()
