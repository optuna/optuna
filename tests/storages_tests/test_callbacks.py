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
    max_retry = 3
    callback = RetryFailedTrialCallback(max_retry=max_retry)
    with StorageSupplier(
        "sqlite", heartbeat_interval=1, grace_period=2, failed_trial_callback=callback
    ) as storage:
        study = optuna.create_study(storage=storage)
        for n_retries in range(1, max_retry + 1):
            # Create a trial and manually set it as failed.
            trial = study.ask({"x": optuna.distributions.FloatDistribution(-1, 1)})
            storage.set_trial_state_values(trial._trial_id, state=TrialState.FAIL)
            callback(study, study.trials[trial.number])  # Manually call callback.

            # Get all trials after another trial for retry is enqueued.
            trials = study.trials
            # Original failed trial + retried trials.
            assert len(trials) == n_retries + 1
            # The last trial before the retried trial must be failed.
            assert trials[trial.number].state == TrialState.FAIL
            # Retry should show the full history of the previous trials.
            assert RetryFailedTrialCallback.retry_history(trials[trial.number + 1]) == list(
                range(n_retries)
            )


def test_retry_history_with_more_than_max_retry() -> None:
    max_retry = 3
    callback = RetryFailedTrialCallback(max_retry=max_retry)
    with StorageSupplier(
        "sqlite", heartbeat_interval=1, grace_period=2, failed_trial_callback=callback
    ) as storage:
        study = optuna.create_study(storage=storage)
        for n_retries in range(max_retry + 2):
            trial = study.ask({"x": optuna.distributions.FloatDistribution(-1, 1)})
            storage.set_trial_state_values(trial._trial_id, state=TrialState.FAIL)
            callback(study, study.trials[trial.number])  # Manually call callback.

        # After max_retry retries, the parameter should be different from the original.
        assert study.trials[0].params["x"] != study.trials[-1].params["x"]


def test_retry_failed_callback_with_exceptions() -> None:
    """Test that RetryFailedTrialCallback is automatically called for exception failures.
    
    This tests the fix for issue #6085 where RetryFailedTrialCallback
    was only working for stale trials, not exception-based failures.
    """
    max_retry = 2
    callback = RetryFailedTrialCallback(max_retry=max_retry)
    
    with StorageSupplier(
        "sqlite", failed_trial_callback=callback
    ) as storage:
        study = optuna.create_study(storage=storage)
        
        def failing_objective(trial):
            # Suggest a parameter (required for valid trial)
            x = trial.suggest_float("x", 0, 1)
            # Always fail with RuntimeError
            raise RuntimeError("Test exception for retry")
        
        # This should automatically trigger the failed_trial_callback
        # when the exception occurs, creating a retry trial
        try:
            study.optimize(failing_objective, n_trials=1, catch=(RuntimeError,))
        except RuntimeError:
            pass
        
        trials = study.trials
        
        # Should have original failed trial + at least one retry trial
        assert len(trials) >= 2, f"Expected at least 2 trials, got {len(trials)}"
        
        # First trial should be FAIL
        failed_trials = [t for t in trials if t.state == TrialState.FAIL]
        assert len(failed_trials) >= 1, "Should have at least one failed trial"
        
        # Should have waiting/retry trials
        waiting_trials = [t for t in trials if t.state == TrialState.WAITING]
        assert len(waiting_trials) >= 1, "Should have at least one retry trial in WAITING state"
        
        # Verify retry metadata is correct
        retry_trial = waiting_trials[0]
        original_trial_number = RetryFailedTrialCallback.retried_trial_number(retry_trial)
        retry_history = RetryFailedTrialCallback.retry_history(retry_trial)
        
        assert original_trial_number is not None, "Retry trial should reference original trial"
        assert len(retry_history) >= 1, "Retry trial should have retry history"
        assert retry_history[0] == failed_trials[0].number, "Retry history should reference failed trial"
