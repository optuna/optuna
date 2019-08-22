"""
Optuna example that optimizes a classifier configuration for the Iris dataset using Dask-ML.

In this example, we optimize a logistic regression classifier configuration for the Iris dataset.
The classifier is from dask-ml while the dataset is from sklearn.
We optimize the choice of solver (admm, gradient descent, or proximal_grad),
the regularization (penalty) when relevant and its strength (C).

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python dask_ml_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize dask_ml_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db

"""

import dask.array as da
from dask_ml.linear_model import LogisticRegression
from dask_ml.model_selection import train_test_split
from distributed import Client
from distributed import LocalCluster
from sklearn.datasets import load_iris


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    iris = load_iris()
    X, y = iris.data, iris.target
    X, y = da.from_array(X, chunks=len(X) // 5), da.from_array(y, chunks=len(y) // 5)

    solver = trial.suggest_categorical('solver', ['admm', 'gradient_descent', 'proximal_grad'])
    C = trial.suggest_uniform('C', 0.0, 1.0)
    max_iter = trial.suggest_uniform(50, 200)

    if solver == 'admm' or solver == 'proximal_grad':
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elastic_net'])
    else:
        # 'penalty' parameter isn't relevant for this solver,
        # so we always specify 'l2' as the dummy value.
        penalty = 'l2'

    classifier = LogisticRegression(max_iter=max_iter, solver=solver, C=C, penalty=penalty)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    classifier.fit(X_train, y_train)

    score = classifier.score(X_test, y_test)
    return score


if __name__ == '__main__':
    import optuna
    study = optuna.create_study(direction='maximize')

    # This is used to initialize the workers that will be used by Dask-ML.
    cluster = LocalCluster(n_workers=2)
    with Client(cluster) as client:
        study.optimize(objective, n_trials=100)

        print('Number of finished trials: ', len(study.trials))

        print('Best trial:')
        trial = study.best_trial

        print('  Value: ', trial.value)

        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))
