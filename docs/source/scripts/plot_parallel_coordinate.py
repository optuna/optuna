import os

import optuna
import plotly


def main():
    def objective(trial):
        x = trial.suggest_uniform('x', -100, 100)
        y = trial.suggest_categorical('y', [-1, 0, 1])
        return x ** 2 + y

    sampler = optuna.samplers.TPESampler(seed=10)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10)

    fig = optuna.visualization.plot_parallel_coordinate(study, params=['x', 'y'])
    fig_html = plotly.offline.plot(fig, output_type='div', include_plotlyjs='cdn', auto_open=False)

    fig_dir = '../plotly_figures'
    os.makedirs(fig_dir, exist_ok=True)
    with open(os.path.join(fig_dir, 'plot_parallel_coordinate.html'), 'w') as f:
        f.write(fig_html)


if __name__ == '__main__':
    main()
