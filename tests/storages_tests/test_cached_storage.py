from unittest.mock import patch

import optuna
from optuna.storages.cached_storage import _CachedStorage
from optuna.storages.cached_storage import RDBStorage
from optuna.trial import TrialState


def test_create_trial() -> None:
    base_storage = RDBStorage("sqlite:///:memory:")
    storage = _CachedStorage(base_storage)
    study_id = storage.create_new_study("test-study")
    frozen_trial = optuna.trial.FrozenTrial(
        number=1,
        state=TrialState.RUNNING,
        value=None,
        datetime_start=None,
        datetime_complete=None,
        params={},
        distributions={},
        user_attrs={},
        system_attrs={},
        intermediate_values={},
        trial_id=1,
    )
    with patch.object(
        base_storage, "_create_new_trial_with_trial", return_value=(1, frozen_trial)
    ):
        storage.create_new_trial(study_id)
    storage.create_new_trial(study_id)


def test_cached_set() -> None:

    """Test CachedStorage does not flush to persistent storages.

     The CachedStorage does not flush when it modifies trial updates of params.

    """

    base_storage = RDBStorage("sqlite:///:memory:")
    storage = _CachedStorage(base_storage)
    study_id = storage.create_new_study("test-study")

    trial_id = storage.create_new_trial(study_id)
    with patch.object(base_storage, "_update_trial", return_value=True) as update_mock:
        with patch.object(base_storage, "set_trial_param", return_value=True) as set_mock:
            storage.set_trial_param(
                trial_id, "paramA", 1.2, optuna.distributions.UniformDistribution(-0.2, 2.3)
            )
            assert update_mock.call_count == 0
            assert set_mock.call_count == 0


def test_uncached_set() -> None:

    """Test CachedStorage does flush to persistent storages.

     The CachedStorage flushes modifications of trials to a persistent storage when
     it modifies either value, intermediate_values, state, user_attrs, or system_attrs.

    """

    base_storage = RDBStorage("sqlite:///:memory:")
    storage = _CachedStorage(base_storage)
    study_id = storage.create_new_study("test-study")

    for state in [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL, TrialState.WAITING]:
        trial_id = storage.create_new_trial(study_id)
        with patch.object(base_storage, "_update_trial", return_value=True) as update_mock:
            with patch.object(base_storage, "set_trial_state", return_value=True) as set_mock:
                storage.set_trial_state(trial_id, state)
                assert update_mock.call_count == 1
                assert set_mock.call_count == 0

    trial_id = storage.create_new_trial(study_id)
    with patch.object(base_storage, "_update_trial", return_value=True) as update_mock:
        with patch.object(base_storage, "set_trial_value", return_value=None) as set_mock:
            storage.set_trial_value(trial_id, 0.3)
            assert update_mock.call_count == 1
            assert set_mock.call_count == 0

    trial_id = storage.create_new_trial(study_id)
    with patch.object(base_storage, "_update_trial", return_value=True) as update_mock:
        with patch.object(
            base_storage, "set_trial_intermediate_value", return_value=None
        ) as set_mock:
            storage.set_trial_intermediate_value(trial_id, 3, 0.3)
            assert update_mock.call_count == 1
            assert set_mock.call_count == 0

    trial_id = storage.create_new_trial(study_id)
    with patch.object(base_storage, "_update_trial", return_value=True) as update_mock:
        with patch.object(base_storage, "set_trial_system_attr", return_value=None) as set_mock:
            storage.set_trial_system_attr(trial_id, "attrA", "foo")
            assert update_mock.call_count == 1
            assert set_mock.call_count == 0

    trial_id = storage.create_new_trial(study_id)
    with patch.object(base_storage, "_update_trial", return_value=True) as update_mock:
        with patch.object(base_storage, "set_trial_user_attr", return_value=None) as set_mock:
            storage.set_trial_user_attr(trial_id, "attrB", "bar")
            assert update_mock.call_count == 1
            assert set_mock.call_count == 0


def test_cache_deletion() -> None:

    """Test CachedStorage deletes a cache when trial finishes."""

    base_storage = RDBStorage("sqlite:///:memory:")
    storage = _CachedStorage(base_storage)
    study_id = storage.create_new_study("test-study")

    for state in [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL, TrialState.WAITING]:
        trial_id = storage.create_new_trial(study_id)
        with patch.object(base_storage, "_update_trial", return_value=True) as update_mock:
            with patch.object(base_storage, "set_trial_state", return_value=True) as set_mock:
                with patch.object(base_storage, "get_trial", return_value=True) as get_mock:
                    storage.get_trial(trial_id)
                    assert update_mock.call_count == 0
                    assert set_mock.call_count == 0
                    assert get_mock.call_count == 0

                    storage.set_trial_state(trial_id, state)
                    assert update_mock.call_count == 1
                    assert set_mock.call_count == 0
                    assert get_mock.call_count == 0

                    storage.get_trial(trial_id)
                    assert update_mock.call_count == 1
                    assert set_mock.call_count == 0
                    assert get_mock.call_count == 1

    trial_id = storage.create_new_trial(study_id)
    with patch.object(base_storage, "_update_trial", return_value=True) as update_mock:
        with patch.object(base_storage, "set_trial_state", return_value=True) as set_mock:
            with patch.object(base_storage, "get_trial", return_value=True) as get_mock:
                storage.get_trial(trial_id)
                assert update_mock.call_count == 0
                assert set_mock.call_count == 0
                assert get_mock.call_count == 0

                storage.set_trial_state(trial_id, TrialState.RUNNING)
                assert update_mock.call_count == 1
                assert set_mock.call_count == 0
                assert get_mock.call_count == 0

                storage.get_trial(trial_id)
                assert update_mock.call_count == 1
                assert set_mock.call_count == 0
                assert get_mock.call_count == 0
