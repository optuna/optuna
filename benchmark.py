from pymoo.indicators.hv import HV
from pymoo.problems.many import WFG1
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

import optuna
from optuna.trial import Trial

n_obj = 10
# k = 2 * (n_obj - 1)
# l = 20
n_var = 20  # k + l**4
problem = WFG1(n_var, n_obj)
reference_point_HV = [2] * n_obj
reference_point_HV[0] = 4
ind = HV(ref_point=reference_point_HV)

laps = 50
step_per_lap = 50


def objective(trial: Trial) -> Sequence[float]:
    x = np.array([trial.suggest_float(f"x{i}", 0, 100) for i in range(n_var)])
    return list(problem.evaluate(x))


# NSGA-II
sampler_II = optuna.samplers.NSGAIISampler()
study = optuna.create_study(directions=["minimize"] * n_obj, sampler=sampler_II)
HVII = []
for _ in range(laps):
    study.optimize(objective, n_trials=step_per_lap)
    pareto_front = study.best_trials
    pareto_front_array = []
    for trial in pareto_front:
        pareto_front_array.append(trial.values)
    pareto_front_array = np.array(pareto_front_array)
    HVII.append(ind(pareto_front_array))


# NSGA-III
reference_point = optuna.samplers.nsgaiii.generate_default_reference_point(n_obj)
sampler_III = optuna.samplers.NSGAIIISampler(reference_point)
study = optuna.create_study(directions=["minimize"] * n_obj, sampler=sampler_III)
HVIII = []
for _ in range(laps):
    study.optimize(objective, n_trials=step_per_lap)
    pareto_front = study.best_trials
    pareto_front_array = []
    for trial in pareto_front:
        pareto_front_array.append(trial.values)
    pareto_front_array = np.array(pareto_front_array)
    HVIII.append(ind(pareto_front_array))

num_trials_axis = [step_per_lap * (i + 1) for i in range(laps)]
plt.plot(num_trials_axis, HVII, label="NSGA-II")
plt.plot(num_trials_axis, HVIII, label="NSGA-III")
plt.title(f"{n_obj} objective, {n_var} variable")
plt.xlabel("trials")
plt.ylabel("hyper volume of pareto front")
plt.legend()
plt.show()
