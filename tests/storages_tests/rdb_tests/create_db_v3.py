from argparse import ArgumentParser
from typing import cast
from typing import Tuple

import optuna


def objective_test_upgrade_v3(trial: optuna.trial.Trial) -> float:
    x1 = trial.suggest_float("x1", -5, 5)
    x2 = trial.suggest_float("x2", -6, 6, step=2)
    x3 = trial.suggest_float("x3", 1e-5, 1e-3, log=True)
    y1 = trial.suggest_int("y1", 0, 10)
    y2 = trial.suggest_int("y2", 1, 20, log=True)
    y3 = trial.suggest_int("y3", 5, 15, step=3)
    z = cast(float, trial.suggest_categorical("z", [-5, 0, 5]))
    trial.set_system_attr("a", 0)
    trial.set_user_attr("b", 1)
    trial.report(0.5, step=0)
    return x1 ** 2 + x2 ** 2 + x3 ** 2 + y1 ** 2 + y2 ** 2 + y3 ** 2 + z ** 2


def mo_objective_test_upgrade_v3(trial: optuna.trial.Trial) -> Tuple[float, float]:
    x1 = trial.suggest_float("x1", -5, 5)
    x2 = trial.suggest_float("x2", -5, 5, step=2)
    x3 = trial.suggest_float("x3", 1.0, 3.0, log=True)
    y1 = trial.suggest_int("y1", 0, 10)
    y2 = trial.suggest_int("y2", 1, 10, log=True)
    y3 = trial.suggest_int("y3", 1, 10, step=3)
    z = cast(float, trial.suggest_categorical("z", [-5, 0, 5]))
    trial.set_system_attr("a", 0)
    trial.set_user_attr("b", 1)
    return x1, x1 ** 2 + x2 ** 2 + x3 ** 2 + y1 ** 2 + y2 ** 2 + y3 ** 2 + z ** 2


if __name__ == "__main__":
    parser = ArgumentParser(description="Create SQLite database for schema upgrade tests.")
    parser.add_argument(
        "--storage-url", default=f"sqlite:///test_upgrade_assets/{optuna.__version__}.db"
    )
    args = parser.parse_args()

    # Create an empty study.
    optuna.create_study(storage=args.storage_url, study_name="single_empty")

    # Create a study for single-objective optimization.
    study = optuna.create_study(storage=args.storage_url, study_name="single")
    study.set_system_attr("c", 2)
    study.set_user_attr("d", 3)
    study.optimize(objective_test_upgrade_v3, n_trials=1)

    # Create a study for multi-objective optimization.
    try:
        optuna.create_study(
            storage=args.storage_url, study_name="multi_empty", directions=["minimize", "minimize"]
        )
        study = optuna.create_study(
            storage=args.storage_url, study_name="multi", directions=["minimize", "minimize"]
        )
        study.set_system_attr("c", 2)
        study.set_user_attr("d", 3)
        study.optimize(mo_objective_test_upgrade_v3, n_trials=1)
    except TypeError:
        print(f"optuna=={optuna.__version__} does not support multi-objective optimization.")

    for s in optuna.get_all_study_summaries(args.storage_url):
        print(f"{s.study_name}, {s.n_trials}")
