"""
Optuna example that runs optimization trials in parallel using Dask.

In this example, we perform hyperparameter optimization on a
RandomForestClassifier which is trained using the handwritten digits
dataset. Trials are run in parallel on a Dask cluster using Optuna's
DaskStorage integration.

To run this example:

    $ python dask_simple.py
"""

from dask.distributed import Client
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import optuna


def objective(trial):
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

    max_depth = trial.suggest_int("max_depth", 2, 10)
    n_estimators = trial.suggest_int("n_estimators", 1, 100)

    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    return score


if __name__ == "__main__":

    with Client() as client:
        print(f"Dask dashboard is available at {client.dashboard_link}")
        storage = optuna.integration.dask.DaskStorage()
        study = optuna.create_study(storage=storage, direction="maximize")
        study.optimize(objective, n_trials=100)

        print(f"Best params: {study.best_params}")
