"""
Optuna example that optimizes a classifier configuration for the Iris dataset using Dask-ML.

In this example, we optimize a logistic regression classifier configuration for the Iris dataset.
The classifier is from dask-ml while the dataset is from sklearn.
We optimize both the choice of solver (admm, gradient descent, or lbfgs) and the regularization strength (C).

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python dask_ml_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize dask_ml_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db

"""

import dask.array as da
from distributed import Client, LocalCluster
from sklearn.datasets import load_iris
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    iris = load_iris()
    X, y = iris.data, iris.target
    X, y = da.from_array(X, chunks=len(X) // 5), da.from_array(y, chunks=len(y) // 5)

    solver = trial.suggest_categorical('solver', ['admm', 'gradient_descent', 'lbfgs'])
    C = trial.suggest_uniform('C', 0.0, 1.0)
    max_iter = 100

    classifier = LogisticRegression(max_iter=max_iter, solver=solver, C=C)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    classifier.fit(X_train, y_train)

    score = classifier.score(X_test, y_test)
    return score


if __name__ == '__main__':
    # This is used to initialize the workers that will be used by Dask-ML
    cluster = LocalCluster(n_workers=4, threads_per_worker=2, local_directory='/tmp')
    client = Client(cluster)
    import optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    print(study.best_trial)
