#!/usr/bin/env python3

"""
Test for the NSGAII Pareto front constraint bug fix.

This test verifies that when all trials are infeasible, best_trials returns
the Pareto front of infeasible trials based on constraint violations.
"""

import sys
import os

# Add the optuna source directory to Python path
sys.path.insert(0, '/tmp/oss-optuna-nsgaii')

import numpy as np
from optuna.study._multi_objective import _get_pareto_front_trials_by_trials
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial, TrialState
from optuna.study._constrained_optimization import _CONSTRAINTS_KEY


def create_mock_trial(trial_number, values, constraints=None):
    """Create a mock FrozenTrial for testing."""
    system_attrs = {}
    if constraints is not None:
        system_attrs[_CONSTRAINTS_KEY] = constraints
    
    return FrozenTrial(
        number=trial_number,
        state=TrialState.COMPLETE,
        values=values,
        params={},
        distributions={},
        user_attrs={},
        system_attrs=system_attrs,
        intermediate_values={},
        trial_id=trial_number,
        datetime_start=None,
        datetime_complete=None
    )


def test_feasible_trials_only():
    """Test case: All trials are feasible (original behavior should be preserved)."""
    print("Test 1: All trials are feasible")
    
    trials = [
        create_mock_trial(0, [1.0, 2.0], [-1.0]),  # feasible
        create_mock_trial(1, [2.0, 1.0], [-1.0]),  # feasible  
        create_mock_trial(2, [1.5, 1.5], [-1.0]),  # feasible, dominated
    ]
    
    directions = [StudyDirection.MINIMIZE, StudyDirection.MINIMIZE]
    result = _get_pareto_front_trials_by_trials(trials, directions, consider_constraint=True)
    
    # Should return the two non-dominated feasible trials
    result_numbers = [t.number for t in result]
    expected = [0, 1]  # trials 0 and 1 are on the Pareto front
    
    print(f"  Expected trials on Pareto front: {expected}")
    print(f"  Actual trials on Pareto front: {result_numbers}")
    assert set(result_numbers) == set(expected), f"Expected {expected}, got {result_numbers}"
    print("  ✅ PASSED")


def test_mixed_feasible_infeasible():
    """Test case: Some trials feasible, some infeasible (should prefer feasible)."""
    print("Test 2: Mixed feasible and infeasible trials")
    
    trials = [
        create_mock_trial(0, [1.0, 2.0], [-1.0]),  # feasible
        create_mock_trial(1, [2.0, 1.0], [1.0]),   # infeasible but good objectives
        create_mock_trial(2, [3.0, 3.0], [-1.0]),  # feasible but dominated
    ]
    
    directions = [StudyDirection.MINIMIZE, StudyDirection.MINIMIZE]
    result = _get_pareto_front_trials_by_trials(trials, directions, consider_constraint=True)
    
    # Should only return feasible trials, ignore infeasible ones
    result_numbers = [t.number for t in result]
    expected = [0]  # only trial 0 is feasible and non-dominated
    
    print(f"  Expected trials on Pareto front: {expected}")
    print(f"  Actual trials on Pareto front: {result_numbers}")
    assert set(result_numbers) == set(expected), f"Expected {expected}, got {result_numbers}"
    print("  ✅ PASSED")


def test_all_infeasible_trials():
    """Test case: All trials are infeasible (this is the bug we're fixing)."""
    print("Test 3: All trials are infeasible")
    
    trials = [
        create_mock_trial(0, [1.0, 2.0], [1.0]),    # infeasible, violation = 1.0
        create_mock_trial(1, [2.0, 1.0], [2.0]),    # infeasible, violation = 2.0  
        create_mock_trial(2, [1.5, 1.5], [1.5]),    # infeasible, violation = 1.5
        create_mock_trial(3, [0.5, 3.0], [1.0]),    # infeasible, violation = 1.0 (same as trial 0)
    ]
    
    directions = [StudyDirection.MINIMIZE, StudyDirection.MINIMIZE]
    result = _get_pareto_front_trials_by_trials(trials, directions, consider_constraint=True)
    
    # Should return trials with smallest constraint violations 
    # Trial 0 and 3 have smallest violation (1.0), so both should be in Pareto front
    result_numbers = [t.number for t in result]
    expected = [0, 3]  # trials with violation = 1.0
    
    print(f"  Expected trials on Pareto front: {expected}")
    print(f"  Actual trials on Pareto front: {result_numbers}")
    
    if len(result) == 0:
        print("  ❌ FAILED: Empty result (this was the original bug)")
        return False
    
    # Verify that all returned trials have the minimum violation
    violations = []
    for trial in result:
        constraints = trial.system_attrs.get(_CONSTRAINTS_KEY)
        violation = sum(max(0, c) for c in constraints)
        violations.append(violation)
    
    print(f"  Violations of returned trials: {violations}")
    
    # All returned trials should have the same (minimum) violation
    min_violation = min(sum(max(0, c) for c in t.system_attrs.get(_CONSTRAINTS_KEY)) 
                       for t in trials)
    expected_violation = 1.0
    
    assert min_violation == expected_violation, f"Expected min violation {expected_violation}, got {min_violation}"
    assert all(v == min_violation for v in violations), f"Not all returned trials have minimum violation"
    
    print("  ✅ PASSED")
    return True


def test_multiple_constraint_violations():
    """Test case: Multiple constraints with different violations."""
    print("Test 4: Multiple constraint violations")
    
    trials = [
        create_mock_trial(0, [1.0, 2.0], [0.5, 0.5]),    # total violation = 1.0
        create_mock_trial(1, [2.0, 1.0], [1.0, 0.0]),    # total violation = 1.0
        create_mock_trial(2, [1.5, 1.5], [1.0, 1.0]),    # total violation = 2.0
        create_mock_trial(3, [0.5, 3.0], [0.2, 0.3]),    # total violation = 0.5 (best)
    ]
    
    directions = [StudyDirection.MINIMIZE, StudyDirection.MINIMIZE]
    result = _get_pareto_front_trials_by_trials(trials, directions, consider_constraint=True)
    
    result_numbers = [t.number for t in result]
    expected = [3]  # trial 3 has the smallest total violation
    
    print(f"  Expected trials on Pareto front: {expected}")
    print(f"  Actual trials on Pareto front: {result_numbers}")
    
    if len(result) == 0:
        print("  ❌ FAILED: Empty result")
        return False
    
    # Check that trial 3 (smallest violation) is returned
    assert 3 in result_numbers, f"Trial 3 with smallest violation should be in result"
    print("  ✅ PASSED")
    return True


def main():
    """Run all tests."""
    print("Testing NSGAII Pareto front constraint bug fix...")
    print("=" * 50)
    
    test_feasible_trials_only()
    print()
    
    test_mixed_feasible_infeasible()
    print()
    
    success = test_all_infeasible_trials()
    print()
    
    test_multiple_constraint_violations()
    print()
    
    if success:
        print("🎉 All tests passed! The fix correctly handles infeasible trials.")
    else:
        print("❌ Some tests failed.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())