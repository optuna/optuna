#!/usr/bin/env python3

"""
Reproduction script for NSGAII Pareto front issue with infeasible trials.
Issue: When all trials are infeasible, best_trials should return Pareto front of 
infeasible trials based on constraint violations, but currently returns empty list.
"""

import optuna
from optuna.samplers import NSGAIISampler


def multi_objective_func(trial):
    """Multi-objective function for optimization."""
    x = trial.suggest_float("x", 0.0, 5.0)
    y = trial.suggest_float("y", 0.0, 5.0)
    
    # Two objectives to optimize
    f1 = x**2 + y**2  # minimize
    f2 = (x - 2)**2 + (y - 2)**2  # minimize
    
    return f1, f2


def impossible_constraints_func(trial):
    """Constraint function that makes ALL trials infeasible."""
    # This constraint is impossible to satisfy for the given parameter ranges
    # All trials will have positive constraint violation
    x = trial.params["x"]
    y = trial.params["y"]
    
    # Impossible constraint: x + y must be negative, but both x and y are >= 0
    constraint_violation = x + y + 1.0  # Always > 0, so always violates constraint
    
    return [constraint_violation]


def main():
    print("Testing NSGAII best_trials with impossible constraints...")
    print("Expected: best_trials should return Pareto front of infeasible trials")
    print("Current behavior: best_trials returns empty list")
    print()
    
    # Create study with NSGAII sampler and constraints
    sampler = NSGAIISampler(
        population_size=20,
        constraints_func=impossible_constraints_func
    )
    
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=sampler
    )
    
    # Run optimization - all trials will be infeasible
    study.optimize(multi_objective_func, n_trials=30)
    
    # Check results
    trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    print(f"Total completed trials: {len(trials)}")
    
    # Check feasibility
    feasible_trials = []
    infeasible_trials = []
    
    for trial in trials:
        constraints = trial.system_attrs.get("constraints")
        if constraints is not None and all(x <= 0.0 for x in constraints):
            feasible_trials.append(trial)
        else:
            infeasible_trials.append(trial)
    
    print(f"Feasible trials: {len(feasible_trials)}")
    print(f"Infeasible trials: {len(infeasible_trials)}")
    
    # Show constraint violations for first few trials
    print("\nConstraint violations for first 5 trials:")
    for i, trial in enumerate(trials[:5]):
        constraints = trial.system_attrs.get("constraints", [])
        print(f"Trial {trial.number}: constraints = {constraints}, total_violation = {sum(max(0, c) for c in constraints)}")
    
    # Check best_trials
    best_trials = study.best_trials
    print(f"\nCurrent best_trials count: {len(best_trials)}")
    
    if len(best_trials) == 0:
        print("❌ ISSUE CONFIRMED: best_trials is empty despite having infeasible trials!")
        print("According to NSGAII constrained domination logic, we should get the")
        print("Pareto front of infeasible trials based on constraint violations.")
        
        # Show what the expected behavior should be
        print("\nExpected behavior: best_trials should contain trials with smallest")
        print("constraint violations that form a Pareto front.")
        
        # Calculate violations to show what should be returned
        violations_and_trials = []
        for trial in infeasible_trials:
            constraints = trial.system_attrs.get("constraints", [])
            total_violation = sum(max(0, c) for c in constraints)
            violations_and_trials.append((total_violation, trial))
        
        violations_and_trials.sort()
        print("\nTrials sorted by constraint violation (smallest first):")
        for i, (violation, trial) in enumerate(violations_and_trials[:5]):
            print(f"  Trial {trial.number}: violation = {violation:.3f}, values = {trial.values}")
    else:
        print("✅ best_trials is not empty")
        for trial in best_trials:
            print(f"  Trial {trial.number}: values = {trial.values}")


if __name__ == "__main__":
    main()