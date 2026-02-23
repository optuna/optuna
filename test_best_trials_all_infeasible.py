"""
Unit test for the NSGAII best_trials constraint bug fix.
This test should be added to tests/study_tests/test_study.py
"""

def test_best_trials_all_infeasible():
    """Test best_trials when all trials are infeasible."""
    import optuna
    from optuna.study._constrained_optimization import _CONSTRAINTS_KEY
    
    study = optuna.create_study(directions=["minimize", "maximize"])
    storage = study._storage

    # Initially, no trials exist
    assert study.best_trials == []

    # Add first infeasible trial with violation = 2.0
    trial = study.ask()
    storage.set_trial_system_attr(trial._trial_id, _CONSTRAINTS_KEY, [2.0])
    study.tell(trial, [1.0, 1.0])

    # Add second infeasible trial with violation = 1.0 (better)
    trial = study.ask()
    storage.set_trial_system_attr(trial._trial_id, _CONSTRAINTS_KEY, [1.0])
    study.tell(trial, [2.0, 2.0])

    # Add third infeasible trial with violation = 1.5
    trial = study.ask()
    storage.set_trial_system_attr(trial._trial_id, _CONSTRAINTS_KEY, [1.5])
    study.tell(trial, [0.5, 0.5])

    # Add fourth infeasible trial with violation = 1.0 (same as second)
    trial = study.ask()
    storage.set_trial_system_attr(trial._trial_id, _CONSTRAINTS_KEY, [1.0])
    study.tell(trial, [1.5, 1.5])

    # When all trials are infeasible, best_trials should return the trials
    # with the smallest constraint violations that form a Pareto front
    best_trials = study.best_trials
    
    # Before the fix: best_trials would be []
    # After the fix: best_trials should contain trials with minimum violation (1.0)
    assert len(best_trials) > 0, "best_trials should not be empty when infeasible trials exist"
    
    # Check that all returned trials have the minimum violation
    min_violation = 1.0
    for trial in best_trials:
        constraints = trial.system_attrs.get(_CONSTRAINTS_KEY)
        total_violation = sum(max(0, c) for c in constraints)
        assert total_violation == min_violation, (
            f"Trial {trial.number} has violation {total_violation}, "
            f"expected {min_violation}"
        )
    
    # Trials 1 and 3 both have violation = 1.0, so both should be in best_trials
    best_trial_numbers = {t.number for t in best_trials}
    expected_numbers = {1, 3}
    assert best_trial_numbers == expected_numbers, (
        f"Expected trials {expected_numbers}, got {best_trial_numbers}"
    )

    print("✅ test_best_trials_all_infeasible passed!")


def test_best_trials_mixed_with_empty_feasible():
    """Test transition from all-infeasible to mixed feasible/infeasible."""
    import optuna
    from optuna.study._constrained_optimization import _CONSTRAINTS_KEY
    
    study = optuna.create_study(directions=["minimize", "maximize"])
    storage = study._storage

    # Start with all infeasible trials
    trial = study.ask()
    storage.set_trial_system_attr(trial._trial_id, _CONSTRAINTS_KEY, [1.0])
    study.tell(trial, [1.0, 1.0])

    trial = study.ask()
    storage.set_trial_system_attr(trial._trial_id, _CONSTRAINTS_KEY, [2.0])
    study.tell(trial, [0.5, 0.5])

    # Should return infeasible Pareto front
    best_trials = study.best_trials
    assert len(best_trials) == 1
    assert best_trials[0].number == 0  # smallest violation

    # Now add a feasible trial
    trial = study.ask()
    storage.set_trial_system_attr(trial._trial_id, _CONSTRAINTS_KEY, [-1.0])  # feasible
    study.tell(trial, [2.0, 2.0])

    # Now should only return feasible trials
    best_trials = study.best_trials
    assert len(best_trials) == 1
    assert best_trials[0].number == 2  # the feasible trial

    print("✅ test_best_trials_mixed_with_empty_feasible passed!")


if __name__ == "__main__":
    print("Running unit tests for best_trials constraint fix...")
    
    try:
        test_best_trials_all_infeasible()
        test_best_trials_mixed_with_empty_feasible()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()