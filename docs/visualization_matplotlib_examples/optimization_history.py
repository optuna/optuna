"""

plot_optimization_history
=========================

.. autofunction:: optuna.visualization.matplotlib.plot_optimization_history

The following code snippet shows how to plot optimization history.

"""

import optuna
import matplotlib.pyplot as plt


def objective(trial):
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_categorical("y", [-1, 0, 1])
    return x**2 + y


sampler = optuna.samplers.TPESampler(seed=10)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=10)

optuna.visualization.matplotlib.plot_optimization_history(study)
plt.tight_layout()

# %%
# .. note::
#     You need to adjust the size of the plot by yourself using ``plt.tight_layout()`` or
#     ``plt.savefig(IMAGE_NAME, bbox_inches='tight')``.
#
# .. seealso::
#     Please refer to :func:`optuna.visualization.plot_optimization_history` for an example.
