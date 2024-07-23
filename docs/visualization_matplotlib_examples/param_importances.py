"""

plot_param_importances
======================

.. autofunction:: optuna.visualization.matplotlib.plot_param_importances

The following code snippet shows how to plot hyperparameter importances.

"""

import optuna


def objective(trial):
    x = trial.suggest_int("x", 0, 2)
    y = trial.suggest_float("y", -1.0, 1.0)
    z = trial.suggest_float("z", 0.0, 1.5)
    return x**2 + y**3 - z**4


sampler = optuna.samplers.RandomSampler(seed=10)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=100)

optuna.visualization.matplotlib.plot_param_importances(study)
