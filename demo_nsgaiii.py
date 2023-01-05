from typing import Sequence

import optuna
from optuna.trial import Trial


def objective2d(trial: Trial) -> Sequence[float]:
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)

    v0 = 4 * x**2 + 4 * y**2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1


def objective3d(trial: Trial) -> Sequence[float]:
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)
    z = trial.suggest_float("z", 0, 3)

    v0 = 2 * x**2 + y**2 + 2 * z**2
    v1 = (x - 3) ** 2 + (y - 1) ** 2 + z**2
    v2 = (x - 2) ** 2 + 2 * (y - 4) ** 2 + 3 * z**2
    return v0, v1, v2


# NSGA-II 2d objective
sampler_II = optuna.samplers.NSGAIISampler()
study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler_II)
study.optimize(objective2d, n_trials=150)

fig = optuna.visualization.plot_pareto_front(study)
fig.update_layout(title="NSGA-II 2D")
fig.show()

# NSGA-III 2d objective
reference_point = optuna.samplers.nsgaiii.generate_default_reference_point(2, 5)
sampler_III = optuna.samplers.NSGAIIISampler(reference_point)
study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler_III)
study.optimize(objective2d, n_trials=150)

fig = optuna.visualization.plot_pareto_front(study)
fig.update_layout(title="NSGA-III 2D")
fig.show()

# NSGA-II 3d objective
sampler_II = optuna.samplers.NSGAIISampler()
study = optuna.create_study(directions=["minimize", "minimize", "minimize"], sampler=sampler_II)
study.optimize(objective3d, n_trials=150)

fig = optuna.visualization.plot_pareto_front(study)
fig.update_layout(title="NSGA-II 3D")
fig.show()

# NSGA-III 3d objective
reference_point = optuna.samplers.nsgaiii.generate_default_reference_point(3, 6)
sampler_III = optuna.samplers.NSGAIIISampler(reference_point)
study = optuna.create_study(directions=["minimize", "minimize", "minimize"], sampler=sampler_III)
study.optimize(objective3d, n_trials=150)

fig = optuna.visualization.plot_pareto_front(study)
fig.update_layout(title="NSGA-III 3D")
fig.show()
