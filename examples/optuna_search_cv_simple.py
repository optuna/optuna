"""
Optuna example that optimizes a classifier configuration using OptunaSearchCV.

In this example, we optimize a classifier configuration for Iris dataset using OptunaSearchCV.
Classifier is from scikit-learn.

We have the following a way to execute this example:

(1) Execute this code directly.
    $ python optuna_search_cv_simple.py

"""

import optuna
from sklearn.datasets import load_iris
from sklearn.svm import SVC

if __name__ == '__main__':
    clf = SVC(gamma='auto')

    param_distributions = {
        'C': optuna.distributions.LogUniformDistribution(1e-10, 1e+10),
        'degree': optuna.distributions.IntUniformDistribution(1, 5),
    }

    optuna_search = optuna.integration.OptunaSearchCV(
        clf,
        param_distributions,
        n_trials=100,
        timeout=600,
        verbose=2,
    )

    X, y = load_iris(return_X_y=True)
    optuna_search.fit(X, y)

    trial = optuna_search.study_.best_trial
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))