"""

plot_intermediate_values
========================

.. autofunction:: optuna.visualization.matplotlib.plot_intermediate_values

The following code snippet shows how to plot intermediate values.

"""

import optuna


def f(x):
    return (x - 2) ** 2


def df(x):
    return 2 * x - 4


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    x = 3
    for step in range(128):
        y = f(x)

        trial.report(y, step=step)
        if trial.should_prune():
            raise optuna.TrialPruned()

        gy = df(x)
        x -= gy * lr

    return y


sampler = optuna.samplers.TPESampler(seed=10)
study = optuna.create_study(sampler=sampler)
study.optimize(objective, n_trials=16)

optuna.visualization.matplotlib.plot_intermediate_values(study)

# %%
# .. note::
#     Please refer to `matplotlib.pyplot.legend
#     <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html>`_
#     to adjust the style of the generated legend.
#
# .. seealso::
#     Please refer to :func:`optuna.visualization.plot_intermediate_values` for an example.
