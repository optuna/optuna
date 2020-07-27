import math
import os

import plotly

import optuna


def ackley(x, y):
    a = 20 * math.exp(-0.2 * math.sqrt(0.5 * (x ** 2 + y ** 2)))
    b = math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y)))
    return -a - b + math.e + 20


def objective(trial, low, high):
    x = trial.suggest_float("x", low, high)
    y = trial.suggest_float("y", low, high)
    return ackley(x, y)


def main():
    sampler = optuna.samplers.RandomSampler(seed=10)

    study0 = optuna.create_study(study_name="x=[0,5), y=[0,5)", sampler=sampler)
    study0.optimize(lambda t: objective(t, 0, 5), n_trials=500)

    study1 = optuna.create_study(study_name="x=[0,4), y=[0,4)", sampler=sampler)
    study1.optimize(lambda t: objective(t, 0, 4), n_trials=500)

    study2 = optuna.create_study(study_name="x=[1,3), y=[1,3)", sampler=sampler)
    study2.optimize(lambda t: objective(t, 1, 3), n_trials=500)

    fig = optuna.visualization.plot_edf([study0, study1, study2])
    fig_html = plotly.offline.plot(fig, output_type="div", include_plotlyjs="cdn", auto_open=False)

    fig_dir = "../plotly_figures"
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(fig_dir, "plot_edf.html"), "w") as f:
        f.write(fig_html)


if __name__ == "__main__":
    main()
