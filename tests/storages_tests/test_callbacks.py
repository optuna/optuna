import optuna
from optuna.storages import RetryFailedTrialCallback
from optuna.testing.storages import StorageSupplier
from optuna.trial import TrialState


def test_retry_history_with_success() -> None:
    # Trial that is not a retry should return empty list.
    with StorageSupplier("sqlite") as storage:
        study = optuna.create_study(storage=storage)
        trial = study.ask()
        study.tell(trial, 1.0)  # Complete the trial to get FrozenTrial.
        assert RetryFailedTrialCallback.retry_history(study.trials[0]) == []


def test_retry_history_with_failures() -> None:
    # Test case 2: First retry should contain only the original trial number
    callback = RetryFailedTrialCallback(max_retry=3)
    with StorageSupplier(
        "sqlite", heartbeat_interval=1, grace_period=2, failed_trial_callback=callback
    ) as storage:
        study = optuna.create_study(storage=storage)

        # Create a trial and manually set it as failed
        trial = study.ask()
        trial.suggest_float("x", -1, 1)
        storage.set_trial_state_values(trial._trial_id, state=TrialState.FAIL)
        callback(study, study.trials[0])  # Manually call callback

        trials = study.trials
        assert len(trials) == 2  # Original failed trial + retried trial
        assert trials[0].state == TrialState.FAIL
        assert RetryFailedTrialCallback.retry_history(trials[1]) == [0]

        # Test case 3: Multiple retries should show the full history
        trial = study.ask()
        trial.suggest_float("x", -1, 1)
        storage.set_trial_state_values(trial._trial_id, state=TrialState.FAIL)
        callback(study, study.trials[1])  # Manually call callback

        trials = study.trials
        assert len(trials) == 3
        assert trials[1].state == TrialState.FAIL
        assert RetryFailedTrialCallback.retry_history(trials[2]) == [0, 1]

        # Test case 4: Verify retry history for a new trial series
        trial = study.ask()
        trial.suggest_float("x", -1, 1)
        storage.set_trial_state_values(trial._trial_id, state=TrialState.FAIL)
        callback(study, study.trials[2])  # Manually call callback

        trials = study.trials
        assert len(trials) == 4
        assert trials[2].state == TrialState.FAIL
        assert RetryFailedTrialCallback.retry_history(trials[3]) == [0, 1, 2]
