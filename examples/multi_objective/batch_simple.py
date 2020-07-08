import numpy as np
import optuna
import optuna._batch_study


def func(x, y):
    v0 = (4 * x) ** 2 + (4 * y) ** 2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1


def objective(trial):
    # Binh and Korn function.
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)

    v0, v1 = func(np.array(x), np.array(y))
    return list(zip(v0.tolist(), v1.tolist()))


batch_size = 4
study = optuna.multi_objective.create_study(["minimize", "minimize"])
bstudy = optuna._batch_study.BatchMultiObjectiveStudy(study, batch_size)
bstudy.batch_optimize(
    objective, n_batches=10,
)

optuna.multi_objective.visualization.plot_pareto_front(study).show()
