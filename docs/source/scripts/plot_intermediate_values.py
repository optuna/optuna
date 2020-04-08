import os
import random

import plotly

import optuna

_seed = 10
_rng = random.Random(_seed)


def f(x):
    return (x - 2) ** 2


def df(x):
    return 2 * x - 4


def objective(trial):
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)

    x = _rng.uniform(0, 3)
    for step in range(32):
        y = f(x)

        trial.report(y, step=step)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        gy = df(x)
        x -= gy * lr

    return y


def main():
    sampler = optuna.samplers.TPESampler(seed=_seed)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10)

    fig = optuna.visualization.plot_intermediate_values(study)
    fig_html = plotly.offline.plot(fig, output_type="div", include_plotlyjs="cdn", auto_open=False)

    fig_dir = "../plotly_figures"
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(fig_dir, "plot_intermediate_values.html"), "w") as f:
        f.write(fig_html)


if __name__ == "__main__":
    main()
