from mock import patch
import pytest
import sys
import tempfile

from optuna.distributions import CategoricalDistribution
from optuna.distributions import json_to_distribution
from optuna.distributions import UniformDistribution
from optuna.storages.rdb.models import SCHEMA_VERSION
from optuna.storages.rdb.models import StudyModel
from optuna.storages.rdb.models import TrialModel
from optuna.storages.rdb.models import TrialParamModel
from optuna.storages.rdb.models import VersionInfoModel
from optuna.storages import RDBStorage
from optuna.structs import DuplicatedStudyError
from optuna.structs import StorageInternalError
from optuna.structs import StudyDirection
from optuna.structs import StudySummary
from optuna.structs import TrialState
from optuna import types
from optuna import version

if types.TYPE_CHECKING:
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
    assert storage.get_all_versions() == ['v0.9.0.a']


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

    trial_id_2_1 = storage.create_new_trial_id(study_id_2)
    trial_id_2_2 = storage.create_new_trial_id(study_id_2)

    storage.set_trial_value(trial_id_2_1, -100)
    storage.set_trial_value(trial_id_2_2, -200)

    storage.set_trial_state(trial_id_2_1, TrialState.COMPLETE)
    storage.set_trial_state(trial_id_2_2, TrialState.COMPLETE)

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


def create_test_storage(enable_cache=True, engine_kwargs=None):
    # type: (bool, Optional[Dict[str, Any]]) -> RDBStorage

    storage = RDBStorage('sqlite:///:memory:',
                         enable_cache=enable_cache,
                         engine_kwargs=engine_kwargs)
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


def test_create_new_trial_number():
    # type: () -> None

    storage = create_test_storage()
    study_id = storage.create_new_study_id()

    trial_id = storage.create_new_trial_id(study_id)
    assert storage._create_new_trial_number(trial_id) == 0

    trial_id = storage.create_new_trial_id(study_id)
    assert storage._create_new_trial_number(trial_id) == 1


def test_update_finished_trial():
    # type: () -> None

    storage = create_test_storage()
    study_id = storage.create_new_study_id()

    # Running trials are allowed to be updated.
    trial_id = storage.create_new_trial_id(study_id)
    assert storage.get_trial(trial_id).state == TrialState.RUNNING

    storage.set_trial_intermediate_value(trial_id, 3, 5)
    storage.set_trial_value(trial_id, 10)
    storage.set_trial_param(trial_id, 'x', 1.5, UniformDistribution(low=1.0, high=2.0))
    storage.set_trial_user_attr(trial_id, 'foo', 'bar')
    storage.set_trial_system_attr(trial_id, 'baz', 'qux')
    storage.set_trial_state(trial_id, TrialState.COMPLETE)

    # Finished trials are not allowed to be updated.
    for state in [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL]:
        trial_id = storage.create_new_trial_id(study_id)
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
            trial_id = storage.create_new_trial_id(study_id)
            storage.set_trial_state(trial_id, state)

        trials = storage.get_all_trials(study_id)
        assert len(trials) == 4

        return trials

    # Storage cache is disabled.
    storage = create_test_storage(enable_cache=False)
    study_id = storage.create_new_study_id()
    trials = setup_trials(storage, study_id)

    with patch.object(
            TrialModel, 'find_or_raise_by_id',
            wraps=TrialModel.find_or_raise_by_id) as mock_object:
        for trial in trials:
            assert storage.get_trial(trial.trial_id) == trial
        assert mock_object.call_count == 4

    with patch.object(TrialModel, 'where_study', wraps=TrialModel.where_study) as mock_object:
        assert storage.get_all_trials(study_id) == trials
        assert mock_object.call_count == 1

    # Storage cache is enabled.
    storage = create_test_storage(enable_cache=True)
    study_id = storage.create_new_study_id()
    trials = setup_trials(storage, study_id)

    with patch.object(
            TrialModel, 'find_or_raise_by_id',
            wraps=TrialModel.find_or_raise_by_id) as mock_object:
        for trial in trials:
            assert storage.get_trial(trial.trial_id) == trial
        assert mock_object.call_count == 1  # Only a running trial was fetched from the storage.

    # If cache is enabled, running trials are fetched from the storage individually.
    with patch.object(TrialModel, 'where_study', wraps=TrialModel.where_study) as mock_object:
        assert storage.get_all_trials(study_id) == trials
        assert mock_object.call_count == 0  # `TrialModel.where_study` has not been called.

    with patch.object(
            TrialModel, 'find_or_raise_by_id',
            wraps=TrialModel.find_or_raise_by_id) as mock_object:
        assert storage.get_all_trials(study_id) == trials
        assert mock_object.call_count == 1
