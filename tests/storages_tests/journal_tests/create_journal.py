"""This script generates assets for testing backward compatibility of `JournalStorage`."""

from argparse import ArgumentParser
import os

import optuna
from optuna.storages.journal import JournalFileBackend


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--storage-url",
        default=f"{os.path.dirname(__file__)}/assets/{optuna.__version__}.log",
    )
    args = parser.parse_args()

    storage = optuna.storages.JournalStorage(JournalFileBackend(args.storage_url))

    # Empty study
    optuna.create_study(storage=storage, study_name="single_empty")

    # Delete the study
    optuna.create_study(storage=storage, study_name="single_to_be_deleted")
    optuna.delete_study(storage=storage, study_name="single_to_be_deleted")

    # Set study user attributes
    study = optuna.create_study(storage=storage, study_name="single_user_attr")
    study.set_user_attr("a", 1)
    study.set_user_attr("b", 2)
    study.set_user_attr("c", 3)

    # Set study system attributes
    study = optuna.create_study(storage=storage, study_name="single_system_attr")
    study.set_system_attr("A", 1)
    study.set_system_attr("B", 2)
    study.set_system_attr("C", 3)

    # Study for single-objective optimization
    def objective(trial: optuna.trial.Trial) -> float:
        x = trial.suggest_float("x", -5, 5)
        y = trial.suggest_int("y", 0, 10)
        z = trial.suggest_categorical("z", [-5, 0, 5])
        trial.report(0.5, step=0)
        trial.set_user_attr(f"a_{trial.number}", 0)
        trial.set_system_attr(f"b_{trial.number}", 1)
        return x**2 + y**2 + z**2

    study = optuna.create_study(storage=storage, study_name="single_optimization")
    study.optimize(objective, n_trials=10)
