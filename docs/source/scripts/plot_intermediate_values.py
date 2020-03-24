import os

import plotly

import optuna


def df(x):
    return 2 * x


def objective(trial):

    next_x = 1  # We start the search at x=1
    gamma = trial.suggest_loguniform('alpha', 1e-5, 1e-1)  # Step size multiplier

    # Stepping through gradient descent to find the minima of x**2
    for step in range(100):
        current_x = next_x
        next_x = current_x - gamma * df(current_x)

        delta = next_x - current_x
        trial.report(current_x, step)

    return delta


def main():
    sampler = optuna.samplers.TPESampler(seed=10)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=5)

    fig = optuna.visualization.plot_intermediate_values(study)
    fig_html = plotly.offline.plot(fig, output_type="div", include_plotlyjs="cdn", auto_open=False)

    fig_dir = "../plotly_figures"
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(fig_dir, "plot_intermediate_values.html"), "w") as f:
        f.write(fig_html)


if __name__ == "__main__":
    main()
