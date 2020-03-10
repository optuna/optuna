import optuna
import plotly


def main():
    def objective(trial):
        x = trial.suggest_uniform('x', -100, 100)
        y = trial.suggest_categorical('y', [-1, 0, 1])
        return x ** 2 + y

    study = optuna.create_study()
    study.optimize(objective, n_trials=10)

    fig = optuna.visualization.plot_contour(study, params=['x', 'y'])
    plotly.offline.plot(fig, filename='../plotly_figures/plot_contour.html',
                        include_plotlyjs=False, auto_open=False)


if __name__ == '__main__':
    main()
