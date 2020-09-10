import optuna


def objective(trial):
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)

    # Binh and Korn function.
    v0 = (4 * x) ** 2 + (4 * y) ** 2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1


def callback(study, trial):
    n = len(study.get_pareto_front_trials())
    print("Trial {}: Pareto front has {} trials.".format(trial.number, n))


if __name__ == "__main__":
    study = optuna.batch.multi_objective.create_study(["minimize", "minimize"])
    study.optimize(objective, n_batches=10, batch_size=4, callbacks=[callback])

    optuna.multi_objective.visualization.plot_pareto_front(study).show()
