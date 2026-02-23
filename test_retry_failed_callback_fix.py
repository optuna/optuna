#!/usr/bin/env python3
"""Test script to verify RetryFailedTrialCallback works with exception-based failures.

This reproduces the issue from #6085 and validates the fix.
"""
import tempfile
import os
import optuna
from optuna.storages import RetryFailedTrialCallback


def test_retry_failed_callback_with_exceptions():
    """Test that RetryFailedTrialCallback retries trials that fail with exceptions."""
    
    with tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False) as f:
        db_path = f.name
    
    try:
        # Create storage with RetryFailedTrialCallback
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{db_path}",
            failed_trial_callback=RetryFailedTrialCallback(max_retry=2),
        )
        
        study = optuna.create_study(
            study_name="test-retry-exceptions",
            storage=storage,
        )
        
        call_count = 0
        
        def failing_objective(trial):
            nonlocal call_count
            call_count += 1
            val = trial.suggest_float("val", 0, 1, step=0.1)
            print(f"Call {call_count}: Trial {trial.number}, val={val}")
            
            # Always fail with RuntimeError
            raise RuntimeError("Test error for retry")
        
        # Run optimization with catch to prevent immediate termination
        try:
            study.optimize(failing_objective, n_trials=1, catch=(RuntimeError,))
        except RuntimeError:
            pass
        
        print(f"Total calls made: {call_count}")
        print(f"Total trials in study: {len(study.trials)}")
        
        # Check the results
        all_trials = study.trials
        failed_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.FAIL]
        waiting_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.WAITING]
        
        print(f"Failed trials: {len(failed_trials)}")
        print(f"Waiting trials (retries): {len(waiting_trials)}")
        
        # Verify retry behavior
        if len(waiting_trials) > 0:
            print("✅ SUCCESS: RetryFailedTrialCallback created retry trials for exception failures!")
            
            # Check retry system attributes
            for trial in waiting_trials:
                retry_info = RetryFailedTrialCallback.retry_history(trial)
                original_trial = RetryFailedTrialCallback.retried_trial_number(trial)
                print(f"  Retry trial {trial.number}: original={original_trial}, history={retry_info}")
        else:
            print("❌ FAILURE: No retry trials created - RetryFailedTrialCallback not working for exceptions")
            
        return len(waiting_trials) > 0
        
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_retry_with_multiple_failures():
    """Test that retries are properly exhausted after max_retry attempts."""
    
    with tempfile.NamedTemporaryFile(suffix='.sqlite3', delete=False) as f:
        db_path = f.name
    
    try:
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{db_path}",
            failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
        )
        
        study = optuna.create_study(
            study_name="test-retry-exhaustion",
            storage=storage,
        )
        
        call_count = 0
        
        def failing_objective(trial):
            nonlocal call_count
            call_count += 1
            print(f"Call {call_count}: Trial {trial.number}")
            raise ValueError("Always fail")
        
        # Run multiple trials to exhaust retries
        for _ in range(5):
            try:
                study.optimize(failing_objective, n_trials=1, catch=(ValueError,))
            except ValueError:
                pass
        
        all_trials = study.trials
        failed_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.FAIL]
        waiting_trials = [t for t in all_trials if t.state == optuna.trial.TrialState.WAITING]
        
        print(f"Total trials: {len(all_trials)}")
        print(f"Failed trials: {len(failed_trials)}")
        print(f"Waiting trials: {len(waiting_trials)}")
        
        # Should have created retry trials up to max_retry limit
        return len(waiting_trials) >= 1 and len(waiting_trials) <= 3
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    print("Testing RetryFailedTrialCallback with exception-based failures...")
    print("=" * 80)
    
    print("\nTest 1: Basic retry functionality")
    success1 = test_retry_failed_callback_with_exceptions()
    
    print("\nTest 2: Retry exhaustion")
    success2 = test_retry_with_multiple_failures()
    
    if success1 and success2:
        print("\n🎉 All tests passed! RetryFailedTrialCallback now works with exceptions.")
    else:
        print("\n💥 Tests failed. RetryFailedTrialCallback still not working properly.")
        exit(1)