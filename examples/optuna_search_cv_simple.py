"""
Optuna example that optimizes a classifier configuration using OptunaSearchCV.

In this example, we optimize a classifier configuration for Iris dataset using OptunaSearchCV.
Classifier is from scikit-learn.

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python optuna_search_cv_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize optuna_search_cv_simple.py objective --n-trials=100 --study \
      $STUDY_NAME --storage sqlite:///example.db

"""

import optuna
from sklearn.datasets import load_iris
from sklearn.svm import SVC

if __name__ == '__main__':
    clf = SVC(gamma='auto')

    param_distributions = {
        'C': optuna.distributions.LogUniformDistribution(1e-10, 1e+10)
    }

    optuna_search = optuna.integration.OptunaSearchCV(
        clf,
        param_distributions,
        verbose=2,
    )

    X, y = load_iris(return_X_y=True)
    optuna_search.fit(X, y)
    print(optuna_search.study_.best_params)
    y_pred = optuna_search.predict(X)
