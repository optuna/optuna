import optuna


def func(x, y):
    # Binh and Korn function.
    v0 = (4 * x) ** 2 + (4 * y) ** 2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1


def objective(trial):
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)

    return func(x, y)


def callback(study, trial):
    n = len(study.get_pareto_front_trials())
    print("Pareto front has {} trials.".format(n))


if __name__ == "__main__":
    study = optuna.multi_objective.create_study(["minimize", "minimize"])
    bstudy = optuna.BatchMultiObjectiveStudy(study, batch_size=4)
    bstudy.batch_optimize(objective, n_batches=10, callbacks=[callback])

    optuna.multi_objective.visualization.plot_pareto_front(study).show()
