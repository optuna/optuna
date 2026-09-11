"""

plot_optimization_history
=========================

.. autofunction:: optuna.visualization.plot_optimization_history

The following code snippet shows how to plot optimization history.

.. _visualization-plot-optimization-history-plotly-content:

"""

# sphinx_gallery_thumbnail_path = "reference/visualization/matplotlib/generated/images/sphx_glr_optuna.visualization.matplotlib.optimization_history_001.png"

import optuna


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x**2 + y


sampler = optuna.samplers.TPESampler(seed=10)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=10)

fig = optuna.visualization.plot_optimization_history(study)
fig
