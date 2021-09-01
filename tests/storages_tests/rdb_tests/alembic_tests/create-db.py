import sys
from typing import cast
from typing import Tuple

import optuna


def objective(trial: optuna.trial.Trial) -> float:
    x = trial.suggest_uniform("x", -5, 5)  # optuna==0.9.0 does not have suggest_float.
    y = trial.suggest_int("y", 0, 10)
    z = cast(float, trial.suggest_categorical("z", [-5, 0, 5]))
    return x ** 2 + y ** 2 + z ** 2


def mo_objective(trial: optuna.trial.Trial) -> Tuple[float, float]:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_int("y", 0, 10)
    z = cast(float, trial.suggest_categorical("z", [-5, 0, 5]))
    return x, x ** 2 + y ** 2 + z ** 2


if __name__ == "__main__":
    storage_url = sys.argv[1]

    # Create an empty study.
    optuna.create_study(storage=storage_url, study_name="single_empty")

    # Create a study for single-objective optimization.
    study = optuna.create_study(storage=storage_url, study_name="single")
    study.optimize(objective, n_trials=1)

    # Create a study for multi-objective optimization.
    try:
        optuna.create_study(
            storage=storage_url, study_name="multi_empty", directions=["minimize", "minimize"]
        )
        study = optuna.create_study(
            storage=storage_url, study_name="multi", directions=["minimize", "minimize"]
        )
        study.optimize(mo_objective, n_trials=1)
    except TypeError:
        print(f"optuna=={optuna.__version__} does not support multi-objective optimization.")

    for s in optuna.get_all_study_summaries(storage_url):
        print(f"{s.study_name}, {s.n_trials}")
