import os

import plotly

import optuna


def objective(trial):
    x = trial.suggest_int("x", 0, 2)
    y = trial.suggest_float("y", -1.0, 1.0)
    z = trial.suggest_float("z", 0.0, 1.5)
    return x ** 2 + y ** 3 - z ** 4


def main():
    sampler = optuna.samplers.RandomSampler(seed=10)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=100)

    fig = optuna.visualization.plot_param_importances(study)
    fig_html = plotly.offline.plot(fig, output_type="div", include_plotlyjs="cdn", auto_open=False)

    fig_dir = "../plotly_figures"
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(fig_dir, "plot_param_importances.html"), "w") as f:
        f.write(fig_html)


if __name__ == "__main__":
    main()
