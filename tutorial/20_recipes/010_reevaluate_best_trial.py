"""
.. _reevaluate_best_trial:

Re-use the best values
==========================

You can re-evaluate the objective with the best hyperparameters again
after the hyperparameter optimization.
For example,

- You investigate deeply the behavior of ML model with the best hyperparameter.
- You have optimized with Optuna using a partial dataset because the training time is long. After the hyperparameter tuning, you train the model using the whole dataset.

Optuna provides interface to re-evaluate the objective function easily.

This tutorial shows an example.

Investigate the best model more
--------------------------------

Let's consider a classical supervised classification problem with Optuna as follows:
"""

from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import optuna


def objective(trial):
    X, y = make_classification(n_features=10, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    C = trial.suggest_loguniform("C", 1e-7, 10.0)

    clf = LogisticRegression(C=C)
    clf.fit(X_train, y_train)

    return clf.score(X_test, y_test)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

print(study.best_trial.value)  # Show the best value.

###################################################################################################
# After the hyperparameter optimization, you are interested in other evaluation metrics
# such as recall, precision, and f1-score on the same dataset.
# You can define another objective function that share most of parts of the ``objective``
# function to reproduce the model with the best hyper-parameters.


def detailed_objective(trial):
    # should be the same code objective if you reproduce the best model
    X, y = make_classification(n_features=10, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    C = trial.suggest_loguniform("C", 1e-7, 10.0)

    clf = LogisticRegression(C=C)
    clf.fit(X_train, y_train)

    # calculate more evaluation metrics
    pred = clf.predict(X_test)

    acc = metrics.accuracy_score(pred, y_test)
    recall = metrics.recall_score(pred, y_test)
    precision = metrics.precision_score(pred, y_test)
    f1 = metrics.f1_score(pred, y_test)

    return acc, f1, recall, precision


###################################################################################################
# Pass ``study.best_trial`` as the argument of ``detailed_objective``.

detailed_objective(study.best_trial)  # calculate acc, f1, recall, and precision

###################################################################################################
# Difference between :class:`~optuna.study.Study.best_trial` and ordinal trials
# ------------------------------------------------------------------------------
#
# You obtain :class:`~optuna.study.Study.best_trial` that returns
# :class:`~optuna.trial.FrozenTrial`.
# The :class:`~optuna.trial.FrozenTrial` is different from an active trail, 
# and behaves differently from :class:`~optuna.trial.Trial` in some situations.
# For example, pruning does not work because :class:`~optuna.trial.FrozenTrial.should_prune`
# always returns ``False``.
#
