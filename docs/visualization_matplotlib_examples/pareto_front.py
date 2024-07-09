"""

plot_pareto_front
=================

.. autofunction:: optuna.visualization.matplotlib.plot_pareto_front

The following code snippet shows how to plot the Pareto front of a study.

"""

import optuna


def objective(trial):
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)

    v0 = 4 * x**2 + 4 * y**2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1


study = optuna.create_study(directions=["minimize", "minimize"])
study.optimize(objective, n_trials=50)

optuna.visualization.matplotlib.plot_pareto_front(study)

# %%
# .. seealso::
#     Please refer to :func:`optuna.visualization.plot_pareto_front` for an example.
