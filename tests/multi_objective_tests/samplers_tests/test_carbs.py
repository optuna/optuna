from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

import optuna
import optuna.multi_objective
from optuna.multi_objective.samplers._carbs import CARBSSampler


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -5, 5)

    cost = abs(x) / 2
    performance = x**2 + y

    return cost, performance


sampler = CARBSSampler()
study = optuna.create_study(directions=['minimize', 'maximize'], sampler=sampler)
study.optimize(objective, n_trials=20)

optuna.visualization.plot_pareto_front(study, target_names=["cost", "performance"])

print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

trial_with_highest_performance = max(study.best_trials, key=lambda t: t.values[1])
print(f"Trial with highest performance: ")
print(f"\tnumber: {trial_with_highest_performance.number}")
print(f"\tparams: {trial_with_highest_performance.params}")
print(f"\tvalues: {trial_with_highest_performance.values}")