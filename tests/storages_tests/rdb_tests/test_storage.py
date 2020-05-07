import pickle
import sys
import tempfile
from unittest.mock import patch

import pytest

from optuna.exceptions import StorageInternalError
from optuna.storages.rdb.models import SCHEMA_VERSION
from optuna.storages.rdb.models import TrialModel
from optuna.storages.rdb.models import VersionInfoModel
from optuna.storages import RDBStorage
from optuna.trial import TrialState
from optuna import type_checking
from optuna import version

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Optional  # NOQA

    from optuna.trial import FrozenTrial  # NOQA


def test_init():
    # type: () -> None

    storage = create_test_storage()
    session = storage.scoped_session()

    version_info = session.query(VersionInfoModel).first()
    assert version_info.schema_version == SCHEMA_VERSION
    assert version_info.library_version == version.__version__

    assert storage.get_current_version() == storage.get_head_version()
    assert storage.get_all_versions() == ["v1.3.0.a", "v1.2.0.a", "v0.9.0.a"]


def test_init_url_template():
    # type: ()-> None

    with tempfile.NamedTemporaryFile(suffix="{SCHEMA_VERSION}") as tf:
        storage = RDBStorage("sqlite:///" + tf.name)
        assert storage.engine.url.database.endswith(str(SCHEMA_VERSION))


def test_init_url_that_contains_percent_character():
    # type: ()-> None

    # Alembic's ini file regards '%' as the special character for variable expansion.
    # We checks `RDBStorage` does not raise an error even if a storage url contains the character.
    with tempfile.NamedTemporaryFile(suffix="%") as tf:
        RDBStorage("sqlite:///" + tf.name)

    with tempfile.NamedTemporaryFile(suffix="%foo") as tf:
        RDBStorage("sqlite:///" + tf.name)

    with tempfile.NamedTemporaryFile(suffix="%foo%%bar") as tf:
        RDBStorage("sqlite:///" + tf.name)


def test_init_db_module_import_error():
    # type: () -> None

    expected_msg = (
        "Failed to import DB access module for the specified storage URL. "
        "Please install appropriate one."
    )

    with patch.dict(sys.modules, {"psycopg2": None}):
        with pytest.raises(ImportError, match=expected_msg):
            RDBStorage("postgresql://user:password@host/database")


def test_engine_kwargs():
    # type: () -> None

    create_test_storage(engine_kwargs={"pool_size": 5})

    with pytest.raises(TypeError):
        create_test_storage(engine_kwargs={"wrong_key": "wrong_value"})


@pytest.mark.parametrize(
    "url,engine_kwargs,expected",
    [
        ("mysql://localhost", {"pool_pre_ping": False}, False),
        ("mysql://localhost", {"pool_pre_ping": True}, True),
        ("mysql://localhost", {}, True),
        ("mysql+pymysql://localhost", {}, True),
        ("mysql://localhost", {"pool_size": 5}, True),
    ],
)
def test_set_default_engine_kwargs_for_mysql(url, engine_kwargs, expected):
    # type: (str, Dict[str, Any], bool)-> None

    RDBStorage._set_default_engine_kwargs_for_mysql(url, engine_kwargs)
    assert engine_kwargs["pool_pre_ping"] is expected


def test_set_default_engine_kwargs_for_mysql_with_other_rdb():
    # type: ()-> None

    # Do not change engine_kwargs if database is not MySQL.
    engine_kwargs = {}  # type: Dict[str, Any]
    RDBStorage._set_default_engine_kwargs_for_mysql("sqlite:///example.db", engine_kwargs)
    assert "pool_pre_ping" not in engine_kwargs
    RDBStorage._set_default_engine_kwargs_for_mysql("postgres:///example.db", engine_kwargs)
    assert "pool_pre_ping" not in engine_kwargs


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

    storage = RDBStorage("sqlite:///:memory:", engine_kwargs=engine_kwargs)
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
    v = VersionInfoModel(version_info_id=1, schema_version=1, library_version="0.0.1")
    session.add(v)
    with pytest.raises(StorageInternalError):
        storage._commit(session)


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
        TrialModel, "find_or_raise_by_id", wraps=TrialModel.find_or_raise_by_id
    ) as mock_object:
        for trial in trials:
            assert storage.get_trial(trial._trial_id) == trial
        assert mock_object.call_count == 1  # Only a running trial was fetched from the storage.

    # Running trials are fetched from the storage individually.
    with patch.object(TrialModel, "where_study", wraps=TrialModel.where_study) as mock_object:
        assert storage.get_all_trials(study_id) == trials
        assert mock_object.call_count == 0  # `TrialModel.where_study` has not been called.

    with patch.object(
        TrialModel, "find_or_raise_by_id", wraps=TrialModel.find_or_raise_by_id
    ) as mock_object:
        assert storage.get_all_trials(study_id) == trials
        assert mock_object.call_count == 1
