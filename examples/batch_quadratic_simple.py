import numpy as np

import optuna
import optuna._batch_study


def f(x, y):
    return (1 - x) ** 2 + y


def objective(btrial):
    # btrial stands for batch trial.
    # It suggests multiple values at once while the original trial returns a single value.
    x = btrial.suggest_float("x", -1, 1)
    y = btrial.suggest_float("y", -1, 1)

    z = f(np.array(x), np.array(y))

    # The return value is expected to be a list of float values.
    return z.tolist()


def callback(study, trial):
    best_trial = study.best_trial
    if best_trial == trial:
        print("best trial is Trial {}".format(trial.number))


batch_size = 4
study = optuna.create_study(direction="maximize")
bstudy = optuna._batch_study.BatchStudy(study, batch_size)
bstudy.batch_optimize(objective, n_batches=10, callbacks=[callback])
