"""
.. _multi_objective_lgbm:

Multi-objective Optimization Example with LightGBM
==================================================

This tutorial is an example of multi-objective optimization with Optuna which optimizes
precision and recall. In practice, it is more common to use F-measure and (ROC-)AUC.

Starting from `v2.4.0 <https://github.com/optuna/optuna/releases/tag/v2.4.0>`_, :class:`~optuna.multi_objective.MultiObjectiveStudy` is made available with :class:`~optuna.study.Study`.
"""


###################################################################################################
# First of all, import requirements.
import lightgbm as lgb
import numpy
from sklearn import datasets
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

import optuna


###################################################################################################
# In the ``objective`` below, we train a model using the Breast cancer dataset before
# calculating precision and recall.
def objective(trial):
    data, target = datasets.load_breast_cancer(return_X_y=True)
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(valid_x)
    pred_labels = numpy.rint(preds)
    precision = precision_score(valid_y, pred_labels)
    recall = recall_score(valid_y, pred_labels)
    return precision, recall


###################################################################################################
# To make :class:`~optuna.study.Study` compatible with multi-objective optimization,
# what we need to do is only to pass the collection of ``"direction"``\\s.

# Add stream handler of stdout to show the messages
import sys
import logging

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

study = optuna.create_study(directions=("maximize", "maximize"))
study.optimize(objective, n_trials=100)


###################################################################################################
# Optuna provides a visualization feature unique to multi-objective: :func:`~optuna.visualization.plot_pareto_front`.
optuna.visualization.plot_pareto_front(study)
