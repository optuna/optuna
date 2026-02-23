"""
Test that demonstrates the fix for issue #6085:
RetryFailedTrialCallback now works with exception-based failures, not just stale trials.

This test shows that when a trial fails with an exception, RetryFailedTrialCallback
is automatically invoked to create retry trials.
"""
from optuna.storages import RetryFailedTrialCallback
from optuna.testing.storages import StorageSupplier
from optuna.trial import TrialState
import optuna


def test_retry_failed_callback_with_exceptions():
    """Test that RetryFailedTrialCallback is automatically called for exception failures."""
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
        
        # Should have original failed trial + one retry trial
        assert len(trials) >= 2, f"Expected at least 2 trials, got {len(trials)}"
        
        # First trial should be FAIL
        failed_trials = [t for t in trials if t.state == TrialState.FAIL]
        assert len(failed_trials) >= 1, "Should have at least one failed trial"
        
        # Should have waiting/retry trials
        waiting_trials = [t for t in trials if t.state == TrialState.WAITING]
        assert len(waiting_trials) >= 1, "Should have at least one retry trial in WAITING state"
        
        # Verify retry metadata
        retry_trial = waiting_trials[0]
        original_trial_number = RetryFailedTrialCallback.retried_trial_number(retry_trial)
        retry_history = RetryFailedTrialCallback.retry_history(retry_trial)
        
        assert original_trial_number is not None, "Retry trial should reference original trial"
        assert len(retry_history) >= 1, "Retry trial should have retry history"
        
        print(f"✅ SUCCESS: Exception-based retry working!")
        print(f"   - Total trials: {len(trials)}")
        print(f"   - Failed trials: {len(failed_trials)}")
        print(f"   - Waiting/retry trials: {len(waiting_trials)}")
        print(f"   - Original trial: {original_trial_number}")
        print(f"   - Retry history: {retry_history}")


def test_retry_exhaustion_with_exceptions():
    """Test that retry limit is respected for exception-based failures."""
    max_retry = 1  # Only allow 1 retry
    callback = RetryFailedTrialCallback(max_retry=max_retry)
    
    with StorageSupplier(
        "sqlite", failed_trial_callback=callback
    ) as storage:
        study = optuna.create_study(storage=storage)
        
        call_count = 0
        def failing_objective(trial):
            nonlocal call_count
            call_count += 1
            trial.suggest_float("x", 0, 1)
            raise ValueError(f"Fail #{call_count}")
        
        # Run optimization multiple times to test retry exhaustion
        for i in range(3):
            try:
                study.optimize(failing_objective, n_trials=1, catch=(ValueError,))
            except ValueError:
                pass
        
        trials = study.trials
        waiting_trials = [t for t in trials if t.state == TrialState.WAITING]
        
        # Should respect max_retry limit
        assert len(waiting_trials) <= max_retry + 1, f"Too many retry trials: {len(waiting_trials)}"
        
        print(f"✅ SUCCESS: Retry limit respected!")
        print(f"   - Total trials: {len(trials)}")
        print(f"   - Waiting trials: {len(waiting_trials)}")
        print(f"   - Max retry: {max_retry}")


if __name__ == "__main__":
    print("Testing RetryFailedTrialCallback fix for issue #6085...")
    print("=" * 60)
    
    try:
        test_retry_failed_callback_with_exceptions()
        print()
        test_retry_exhaustion_with_exceptions()
        print("\n🎉 All tests passed! Fix is working correctly.")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)