from __future__ import annotations

import itertools
import multiprocessing
import time
from typing import Any
from unittest.mock import Mock
from unittest.mock import patch
import warnings

import pytest

import optuna
from optuna import Study
from optuna.storages import RDBStorage
from optuna.storages._callbacks import RetryFailedTrialCallback
from optuna.storages._heartbeat import BaseHeartbeat
from optuna.storages._heartbeat import is_heartbeat_enabled
from optuna.testing.storages import STORAGE_MODES_HEARTBEAT
from optuna.testing.storages import StorageSupplier
from optuna.testing.threading import _TestableThread
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


@pytest.mark.parametrize("storage_mode", STORAGE_MODES_HEARTBEAT)
def test_repeatedly_called_record_heartbeat(storage_mode: str) -> None:
    heartbeat_interval = 1
    grace_period = 2

    with StorageSupplier(
        storage_mode, heartbeat_interval=heartbeat_interval, grace_period=grace_period
    ) as storage:
        assert is_heartbeat_enabled(storage)
        assert isinstance(storage, BaseHeartbeat)

        study1 = optuna.create_study(storage=storage)

        with pytest.warns(UserWarning):
            trial1 = study1.ask()
        storage.record_heartbeat(trial1._trial_id)
        storage.record_heartbeat(trial1._trial_id)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES_HEARTBEAT)
def test_fail_stale_trials_with_optimize(storage_mode: str) -> None:
    heartbeat_interval = 1
    grace_period = 2

    with StorageSupplier(
        storage_mode, heartbeat_interval=heartbeat_interval, grace_period=grace_period
    ) as storage:
        assert is_heartbeat_enabled(storage)
        assert isinstance(storage, BaseHeartbeat)

        study1 = optuna.create_study(storage=storage)
        study2 = optuna.create_study(storage=storage)

        with pytest.warns(UserWarning):
            trial1 = study1.ask()
            trial2 = study2.ask()
        storage.record_heartbeat(trial1._trial_id)
        storage.record_heartbeat(trial2._trial_id)
        time.sleep(grace_period + 1)

        assert study1.trials[0].state is TrialState.RUNNING
        assert study2.trials[0].state is TrialState.RUNNING

        # Exceptions raised in spawned threads are caught by `_TestableThread`.
        with patch("optuna.storages._heartbeat.Thread", _TestableThread):
            study1.optimize(lambda _: 1.0, n_trials=1)

        assert study1.trials[0].state is TrialState.FAIL  # type: ignore [comparison-overlap]
        assert study2.trials[0].state is TrialState.RUNNING


@pytest.mark.parametrize("storage_mode", STORAGE_MODES_HEARTBEAT)
def test_invalid_heartbeat_interval_and_grace_period(storage_mode: str) -> None:
    with pytest.raises(ValueError):
        with StorageSupplier(storage_mode, heartbeat_interval=-1):
            pass

    with pytest.raises(ValueError):
        with StorageSupplier(storage_mode, grace_period=-1):
            pass


@pytest.mark.parametrize("storage_mode", STORAGE_MODES_HEARTBEAT)
def test_failed_trial_callback(storage_mode: str) -> None:
    heartbeat_interval = 1
    grace_period = 2

    def _failed_trial_callback(study: Study, trial: FrozenTrial) -> None:
        assert study._storage.get_study_system_attrs(study._study_id)["test"] == "A"
        assert trial.system_attrs["test"] == "B"

    failed_trial_callback = Mock(wraps=_failed_trial_callback)

    with StorageSupplier(
        storage_mode,
        heartbeat_interval=heartbeat_interval,
        grace_period=grace_period,
        failed_trial_callback=failed_trial_callback,
    ) as storage:
        assert is_heartbeat_enabled(storage)
        assert isinstance(storage, BaseHeartbeat)

        study = optuna.create_study(storage=storage)
        study._storage.set_study_system_attr(study._study_id, "test", "A")

        with pytest.warns(UserWarning):
            trial = study.ask()
        trial.storage.set_trial_system_attr(trial._trial_id, "test", "B")
        storage.record_heartbeat(trial._trial_id)
        time.sleep(grace_period + 1)

        # Exceptions raised in spawned threads are caught by `_TestableThread`.
        with patch("optuna.storages._heartbeat.Thread", _TestableThread):
            study.optimize(lambda _: 1.0, n_trials=1)
            failed_trial_callback.assert_called_once()


@pytest.mark.parametrize(
    "storage_mode,max_retry", itertools.product(STORAGE_MODES_HEARTBEAT, [None, 0, 1])
)
def test_retry_failed_trial_callback(storage_mode: str, max_retry: int | None) -> None:
    heartbeat_interval = 1
    grace_period = 2

    with StorageSupplier(
        storage_mode,
        heartbeat_interval=heartbeat_interval,
        grace_period=grace_period,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=max_retry),
    ) as storage:
        assert is_heartbeat_enabled(storage)
        assert isinstance(storage, BaseHeartbeat)

        study = optuna.create_study(storage=storage)

        with pytest.warns(UserWarning):
            trial = study.ask()
        trial.suggest_float("_", -1, -1)
        trial.report(0.5, 1)
        storage.record_heartbeat(trial._trial_id)
        time.sleep(grace_period + 1)

        # Exceptions raised in spawned threads are caught by `_TestableThread`.
        with patch("optuna.storages._heartbeat.Thread", _TestableThread):
            study.optimize(lambda _: 1.0, n_trials=1)

        # Test the last trial to see if it was a retry of the first trial or not.
        # Test max_retry=None to see if trial is retried.
        # Test max_retry=0 to see if no trials are retried.
        # Test max_retry=1 to see if trial is retried.
        assert RetryFailedTrialCallback.retried_trial_number(study.trials[1]) == (
            None if max_retry == 0 else 0
        )
        # Test inheritance of trial fields.
        if max_retry != 0:
            assert study.trials[0].params == study.trials[1].params
            assert study.trials[0].distributions == study.trials[1].distributions
            assert study.trials[0].user_attrs == study.trials[1].user_attrs
            # Only `intermediate_values` are not inherited.
            assert study.trials[1].intermediate_values == {}


@pytest.mark.parametrize(
    "storage_mode,max_retry", itertools.product(STORAGE_MODES_HEARTBEAT, [None, 0, 1])
)
def test_retry_failed_trial_callback_intermediate(
    storage_mode: str, max_retry: int | None
) -> None:
    heartbeat_interval = 1
    grace_period = 2

    with StorageSupplier(
        storage_mode,
        heartbeat_interval=heartbeat_interval,
        grace_period=grace_period,
        failed_trial_callback=RetryFailedTrialCallback(
            max_retry=max_retry, inherit_intermediate_values=True
        ),
    ) as storage:
        assert is_heartbeat_enabled(storage)
        assert isinstance(storage, BaseHeartbeat)

        study = optuna.create_study(storage=storage)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            trial = study.ask()
        trial.suggest_float("_", -1, -1)
        trial.report(0.5, 1)
        storage.record_heartbeat(trial._trial_id)
        time.sleep(grace_period + 1)

        # Exceptions raised in spawned threads are caught by `_TestableThread`.
        with patch("optuna.storages._heartbeat.Thread", _TestableThread):
            study.optimize(lambda _: 1.0, n_trials=1)

        # Test the last trial to see if it was a retry of the first trial or not.
        # Test max_retry=None to see if trial is retried.
        # Test max_retry=0 to see if no trials are retried.
        # Test max_retry=1 to see if trial is retried.
        assert RetryFailedTrialCallback.retried_trial_number(study.trials[1]) == (
            None if max_retry == 0 else 0
        )
        # Test inheritance of trial fields.
        if max_retry != 0:
            assert study.trials[0].params == study.trials[1].params
            assert study.trials[0].distributions == study.trials[1].distributions
            assert study.trials[0].user_attrs == study.trials[1].user_attrs
            assert study.trials[0].intermediate_values == study.trials[1].intermediate_values


@pytest.mark.parametrize("grace_period", [None, 2])
def test_fail_stale_trials(grace_period: int | None) -> None:
    storage_mode = "sqlite"
    heartbeat_interval = 1
    _grace_period = (heartbeat_interval * 2) if grace_period is None else grace_period

    def failed_trial_callback(study: "optuna.Study", trial: FrozenTrial) -> None:
        assert study._storage.get_study_system_attrs(study._study_id)["test"] == "A"
        assert trial.system_attrs["test"] == "B"

    def check_change_trial_state_to_fail(study: "optuna.Study") -> None:
        assert study.trials[0].state is TrialState.RUNNING
        optuna.storages.fail_stale_trials(study)
        assert study.trials[0].state is TrialState.FAIL  # type: ignore [comparison-overlap]

    def check_keep_trial_state_in_running(study: "optuna.Study") -> None:
        assert study.trials[0].state is TrialState.RUNNING
        optuna.storages.fail_stale_trials(study)
        assert study.trials[0].state is TrialState.RUNNING

    with StorageSupplier(storage_mode) as storage:
        assert isinstance(storage, RDBStorage)
        storage.heartbeat_interval = heartbeat_interval
        storage.grace_period = grace_period
        storage.failed_trial_callback = failed_trial_callback
        study = optuna.create_study(storage=storage)
        study._storage.set_study_system_attr(study._study_id, "test", "A")

        with pytest.warns(UserWarning):
            trial = study.ask()
        trial.storage.set_trial_system_attr(trial._trial_id, "test", "B")

        time.sleep(_grace_period + 1)
        check_keep_trial_state_in_running(study)

        storage.record_heartbeat(trial._trial_id)

        check_keep_trial_state_in_running(study)

        time.sleep(_grace_period + 1)
        check_change_trial_state_to_fail(study)


def run_fail_stale_trials(storage_url: str, sleep_time: int) -> None:
    heartbeat_interval = 1
    grace_period = 2

    storage = RDBStorage(storage_url)
    storage.heartbeat_interval = heartbeat_interval
    storage.grace_period = grace_period

    original_set_trial_state_values = storage.set_trial_state_values

    def _set_trial_state_values(*args: Any, **kwargs: Any) -> bool:
        # The second process fails to set state due to the race condition.
        time.sleep(sleep_time)
        return original_set_trial_state_values(*args, **kwargs)

    storage.set_trial_state_values = _set_trial_state_values  # type: ignore[method-assign]

    study = optuna.load_study(study_name=None, storage=storage)
    optuna.storages.fail_stale_trials(study)


def test_fail_stale_trials_with_race_condition() -> None:
    grace_period = 2
    storage_mode = "sqlite"

    with StorageSupplier(storage_mode) as storage:
        assert isinstance(storage, RDBStorage)

        study = optuna.create_study(storage=storage)

        trial = study.ask()
        storage.record_heartbeat(trial._trial_id)
        time.sleep(grace_period + 1)
        p1 = multiprocessing.Process(target=run_fail_stale_trials, args=(storage.url, 1))
        p1.start()
        p2 = multiprocessing.Process(target=run_fail_stale_trials, args=(storage.url, 2))
        p2.start()
        p1.join()
        p2.join()
        assert p1.exitcode == 0
        assert p2.exitcode == 0
        assert study.trials[0].state is TrialState.FAIL


def test_get_stale_trial_ids() -> None:
    storage_mode = "sqlite"
    heartbeat_interval = 1
    grace_period = 2

    with StorageSupplier(storage_mode) as storage:
        assert isinstance(storage, RDBStorage)
        storage.heartbeat_interval = heartbeat_interval
        storage.grace_period = grace_period
        study = optuna.create_study(storage=storage)

        with pytest.warns(UserWarning):
            trial = study.ask()
        storage.record_heartbeat(trial._trial_id)
        time.sleep(grace_period + 1)
        assert len(storage._get_stale_trial_ids(study._study_id)) == 1
        assert storage._get_stale_trial_ids(study._study_id)[0] == trial._trial_id


@pytest.mark.parametrize("storage_mode", STORAGE_MODES_HEARTBEAT)
def test_retry_failed_trial_callback_repetitive_failure(storage_mode: str) -> None:
    heartbeat_interval = 1
    grace_period = 2
    max_retry = 3
    n_trials = 5

    with StorageSupplier(
        storage_mode,
        heartbeat_interval=heartbeat_interval,
        grace_period=grace_period,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=max_retry),
    ) as storage:
        assert is_heartbeat_enabled(storage)
        assert isinstance(storage, BaseHeartbeat)

        study = optuna.create_study(storage=storage)

        # Make repeatedly failed and retried trials by heartbeat.
        for _ in range(n_trials):
            with pytest.warns(UserWarning):
                trial = study.ask()
            storage.record_heartbeat(trial._trial_id)
            time.sleep(grace_period + 1)
            optuna.storages.fail_stale_trials(study)

        trials = study.trials

        assert len(trials) == n_trials + 1

        assert "failed_trial" not in trials[0].system_attrs
        assert "retry_history" not in trials[0].system_attrs

        # The trials 1-3 are retried ones originating from the trial 0.
        assert trials[1].system_attrs["failed_trial"] == 0
        assert trials[1].system_attrs["retry_history"] == [0]

        assert trials[2].system_attrs["failed_trial"] == 0
        assert trials[2].system_attrs["retry_history"] == [0, 1]

        assert trials[3].system_attrs["failed_trial"] == 0
        assert trials[3].system_attrs["retry_history"] == [0, 1, 2]

        # Trials 4 and later are the newly started ones and
        # they are retried after exceeding max_retry.
        assert "failed_trial" not in trials[4].system_attrs
        assert "retry_history" not in trials[4].system_attrs
        assert trials[5].system_attrs["failed_trial"] == 4
        assert trials[5].system_attrs["retry_history"] == [4]
