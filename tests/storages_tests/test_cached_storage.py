from unittest.mock import patch

import pytest

import optuna
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._cached_storage import RDBStorage
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
    with patch.object(base_storage, "_create_new_trial", return_value=frozen_trial):
        storage.create_new_trial(study_id)
    storage.create_new_trial(study_id)


def test_set_trial_state_values() -> None:
    base_storage = RDBStorage("sqlite:///:memory:")
    storage = _CachedStorage(base_storage)
    study_id = storage.create_new_study("test-study")
    trial_id = storage.create_new_trial(study_id)
    storage.set_trial_state_values(trial_id, state=TrialState.COMPLETE)

    cached_trial = storage.get_trial(trial_id)
    base_trial = base_storage.get_trial(trial_id)

    assert cached_trial == base_trial


def test_uncached_set() -> None:

    """Test CachedStorage does flush to persistent storages.

    The CachedStorage flushes any modification of trials to a persistent storage immediately.

    """

    base_storage = RDBStorage("sqlite:///:memory:")
    storage = _CachedStorage(base_storage)
    study_id = storage.create_new_study("test-study")

    trial_id = storage.create_new_trial(study_id)
    trial = storage.get_trial(trial_id)
    with patch.object(base_storage, "set_trial_state_values", return_value=True) as set_mock:
        storage.set_trial_state_values(trial_id, state=trial.state, values=(0.3,))
        assert set_mock.call_count == 1

    trial_id = storage.create_new_trial(study_id)
    with patch.object(
        base_storage, "_check_and_set_param_distribution", return_value=True
    ) as set_mock:
        storage.set_trial_param(
            trial_id, "paramA", 1.2, optuna.distributions.FloatDistribution(-0.2, 2.3)
        )
        assert set_mock.call_count == 1

    trial_id = storage.create_new_trial(study_id)
    with patch.object(base_storage, "set_trial_param", return_value=True) as set_mock:
        storage.set_trial_param(
            trial_id, "paramA", 1.2, optuna.distributions.FloatDistribution(-0.2, 2.3)
        )
        assert set_mock.call_count == 1

    for state in [TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL, TrialState.WAITING]:
        trial_id = storage.create_new_trial(study_id)
        with patch.object(base_storage, "set_trial_state_values", return_value=True) as set_mock:
            storage.set_trial_state_values(trial_id, state=state)
            assert set_mock.call_count == 1

    trial_id = storage.create_new_trial(study_id)
    with patch.object(base_storage, "set_trial_intermediate_value", return_value=None) as set_mock:
        storage.set_trial_intermediate_value(trial_id, 3, 0.3)
        assert set_mock.call_count == 1

    trial_id = storage.create_new_trial(study_id)
    with patch.object(base_storage, "set_trial_system_attr", return_value=None) as set_mock:
        storage.set_trial_system_attr(trial_id, "attrA", "foo")
        assert set_mock.call_count == 1

    trial_id = storage.create_new_trial(study_id)
    with patch.object(base_storage, "set_trial_user_attr", return_value=None) as set_mock:
        storage.set_trial_user_attr(trial_id, "attrB", "bar")
        assert set_mock.call_count == 1


def test_read_trials_from_remote_storage() -> None:

    base_storage = RDBStorage("sqlite:///:memory:")
    storage = _CachedStorage(base_storage)
    study_id = storage.create_new_study("test-study")

    storage.read_trials_from_remote_storage(study_id)

    # Non-existent study.
    with pytest.raises(KeyError):
        storage.read_trials_from_remote_storage(study_id + 1)
