import os

import plotly

import optuna


def objective(trial):
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)

    v0 = (4 * x) ** 2 + (4 * y) ** 2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1


def main():
    study = optuna.multi_objective.create_study(["minimize", "minimize"])
    study.optimize(objective, n_trials=50)

    fig = optuna.multi_objective.visualization.plot_pareto_front(study)
    fig_html = plotly.offline.plot(fig, output_type="div", include_plotlyjs="cdn", auto_open=False)

    fig_dir = "../plotly_figures"
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(fig_dir, "plot_pareto_front.html"), "w") as f:
        f.write(fig_html)


if __name__ == "__main__":
    main()
