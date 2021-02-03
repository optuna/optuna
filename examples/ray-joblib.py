"""
Optuna example that optimizes a classifier configuration for the Iris dataset using Ray with
joblib backend.

In this example, we optimize a classifier configuration for Iris dataset. Classifiers are from
scikit-learn. We optimize both the choice of classifier (among SVC and RandomForest) and their
hyper parameters.

"""
import logging

import joblib
import ray
from ray.util.joblib import register_ray
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm

import optuna


# Get a URL of the Ray dashboard.
try:
    ray.init(address="auto")
except ConnectionError:
    ray.init()
# Disable the warning to suppress the log.
ray.init(logging_level=logging.ERROR, ignore_reinit_error=True)
register_ray()


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = sklearn.ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=10
        )

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    with joblib.parallel_backend("ray", n_jobs=-1):
        study.optimize(objective, n_trials=100)

    print(f"Number of finished trials: {len(study.trials)}")

    print(f"Elapsed time: {study.trials[-1].datetime_complete - study.trials[0].datetime_start}")

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
