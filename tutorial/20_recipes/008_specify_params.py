"""
.. _specify_params:

Specify Hyperparameters
=======================

Sometimes, it is natural for you to try some experiments with your out-of-box hyperparameters.
For example, you have some specific sets of hyperparameters to try in your mind before using Optuna for the best hyperparameters.

First scenario
--------------

Try some sets of hyperparameters with :func:`optuna.study.Study.enqueue_trial`.
"""

import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import optuna


###################################################################################################
# Define the objective function.
def objective(trial):
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)
    dvalid = lgb.Dataset(valid_x, label=valid_y)

    param = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0+1e-12),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    # Add a callback for pruning.
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "auc")
    gbm = lgb.train(
        param, dtrain, valid_sets=[dvalid], verbose_eval=False, callbacks=[pruning_callback]
    )

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(valid_y, pred_labels)
    return accuracy


###################################################################################################
# When You have some sets of hyperparameters that you want to try,
# :func:`~optuna.study.Study.enqueue_trial` does the thing.

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

###################################################################################################
# First, try the default hyperparameters and try some larger ``"bagging_fraq"`` value.

study.enqueue_trial(
    {
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "min_child_sample": 20,
    }
)

study.enqueue_trial(
    {
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "min_child_sample": 20,
    }
)

import logging
import sys

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study.optimize(objective, n_trials=100, timeout=600)

###################################################################################################
# Second scenario
# ---------------
#
# You have tried some sets of hyperparameters manually and then you want to have Optuna find better sets of hyperparameters.
# If this is the case, :func:`optuna.study.Study.add_trial` plays the role.

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.add_trial(optuna.trial.create_trial(
    params={
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
    },
    distributions={
        "bagging_fraction": optuna.distributions.UniformDistribution(0.4, 1.0+1e-12),
        "bagging_freq": optuna.distributions.IntUniformDistribution(0, 7),
    },
    value=0.94,
))
study.add_trial(optuna.trial.create_trial(
    params={
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
    },
    distributions={
        "bagging_fraction": optuna.distributions.UniformDistribution(0.4, 1.0+1e-12),
        "bagging_freq": optuna.distributions.IntUniformDistribution(0, 7),
    },
    value=0.95,
))
study.optimize(objective, n_trials=100, timeout=600)
