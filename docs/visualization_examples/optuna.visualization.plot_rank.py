"""

plot_rank
=========

.. autofunction:: optuna.visualization.plot_rank

The following code snippet shows how to plot the parameter relationship as a rank plot.

"""

import optuna
from plotly.io import show


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])

    c0 = 400 - (x + y) ** 2
    trial.set_constraint("c0", c0)

    return x**2 + y


sampler = optuna.samplers.TPESampler(seed=10)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=30)

fig = optuna.visualization.plot_rank(study, params=["x", "y"])
show(fig)
