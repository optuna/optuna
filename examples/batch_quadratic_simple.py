import optuna


def objective(batch_trial):
    # batch_trial suggests multiple values at once.
    x = batch_trial.suggest_float("x", -1, 1)
    y = batch_trial.suggest_float("y", -1, 1)

    z = (1 - x) ** 2 + y

    # The return value is expected to be a numpy array of float values.
    return z


def callback(study, trial):
    best_trial = study.best_trial
    if best_trial == trial:
        print("best trial is Trial {}".format(trial.number))


if __name__ == "__main__":
    study = optuna.batch.create_study(direction="maximize", batch_size=4)
    study.optimize(objective, n_batches=10, callbacks=[callback])
