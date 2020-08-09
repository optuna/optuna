import optuna


def objective(trial):
    # Binh and Korn function.
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)

    v0 = (4 * x) ** 2 + (4 * y) ** 2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1


study = optuna.multi_objective.create_study(["minimize", "minimize"])
study.optimize(objective, n_trials=100)

optuna.multi_objective.visualization.plot_pareto_front(study).show()
