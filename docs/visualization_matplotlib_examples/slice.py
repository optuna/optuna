"""

plot_slice
============

.. autofunction:: optuna.visualization.matplotlib.plot_slice

The following code snippet shows how to plot the parameter relationship as slice plot.

"""

import optuna


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x**2 + y


sampler = optuna.samplers.TPESampler(seed=10)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=10)

optuna.visualization.matplotlib.plot_slice(study, params=["x", "y"])

# %%
# .. seealso::
#     Please refer to :func:`optuna.visualization.plot_slice` for an example.
