import optuna


def f(x, y):
    return (1 - x) ** 2 + y


def objective(btrial):
    # btrial stands for batch trial.
    # It suggests multiple values at once while the original trial returns a single value.
    x = btrial.suggest_float("x", -1, 1)
    y = btrial.suggest_float("y", -1, 1)

    z = f(x, y)

    # The return value is expected to be a numpy array of float values.
    return z


def callback(study, trial):
    best_trial = study.best_trial
    if best_trial == trial:
        print("best trial is Trial {}".format(trial.number))


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    bstudy = optuna.BatchStudy(study, batch_size=4)
    bstudy.batch_optimize(objective, n_batches=10, callbacks=[callback])
