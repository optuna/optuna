"""

plot_contour
============

.. autofunction:: optuna.visualization.matplotlib.plot_contour

The following code snippet shows how to plot the parameter relationship as contour plot.

"""

import optuna


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x**2 + y


sampler = optuna.samplers.TPESampler(seed=10)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=30)

optuna.visualization.matplotlib.plot_contour(study, params=["x", "y"])

# %%
# .. seealso::
#     Please refer to :func:`optuna.visualization.plot_contour` for an example.
