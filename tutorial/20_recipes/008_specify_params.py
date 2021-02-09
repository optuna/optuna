"""
.. _specify_params:

Specify Hyperparameters Manually
================================

It's natural that you have some specific sets of hyperparameters to try first such as initial learning rate
values and the number of leaves.
Also, it's also possible that you've already tried those sets before having Optuna find better
sets of hyperparameters.

Optuna provides two APIs to support such cases:

1. Passing those sets of hyperparameters and let Optuna evaluate them - :func:`~optuna.study.Study.enqueue_trial`
2. Adding the results of those sets as completed ``Trial``\\s - :func:`~optuna.study.Study.add_trial`

First Scenario: Have Optuna evaluate your hyperparameters
---------------------------------------------------------

In this scenario, let's assume you have some out-of-box sets of hyperparameters but have not
evaluated them yet and decided to use Optuna to find better sets of hyperparameters.

Optuna has :func:`optuna.study.Study.enqueue_trial` which lets you pass those sets of
hyperparameters to Optuna and Optuna will evaluate them.

This section walks you through how to use this lit API with `LightGBM <https://lightgbm.readthedocs.io/en/latest/>`_.
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
        "bagging_fraction": min(trial.suggest_float("bagging_fraction", 0.4, 1.0 + 1e-12), 1),
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
# Then, construct ``Study`` for hyperparameter optimization.

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())

###################################################################################################
# Here, we get Optuna evaluate some sets with larger ``"bagging_fraq"`` value and
# the default values.

study.enqueue_trial(
    {
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "min_child_samples": 20,
    }
)

study.enqueue_trial(
    {
        "bagging_fraction": 0.75,
        "bagging_freq": 5,
        "min_child_samples": 20,
    }
)

import logging
import sys

# Add stream handler of stdout to show the messages to see Optuna works expectedly.
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study.optimize(objective, n_trials=100, timeout=600)

###################################################################################################
# Second scenario: Have Optuna utilize already evaluated hyperparameters
# ----------------------------------------------------------------------
#
# In this scenario, let's assume you have some out-of-box sets of hyperparameters and
# you have already evaluated them but the results are not desirable so that you are thinking of
# using Optuna.
#
# Optuna has :func:`optuna.study.Study.add_trial` which lets you register those results
# to Optuna and then Optuna will sample hyperparameters taking them into account.
#
# In this section,  the ``objective`` is the same as the first scenario.

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.add_trial(
    optuna.trial.create_trial(
        params={
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "min_child_samples": 20,
        },
        distributions={
            "bagging_fraction": optuna.distributions.UniformDistribution(0.4, 1.0 + 1e-12),
            "bagging_freq": optuna.distributions.IntUniformDistribution(0, 7),
            "min_child_samples": optuna.distributions.IntUniformDistribution(5, 100),
        },
        value=0.94,
    )
)
study.add_trial(
    optuna.trial.create_trial(
        params={
            "bagging_fraction": 0.75,
            "bagging_freq": 5,
            "min_child_samples": 20,
        },
        distributions={
            "bagging_fraction": optuna.distributions.UniformDistribution(0.4, 1.0 + 1e-12),
            "bagging_freq": optuna.distributions.IntUniformDistribution(0, 7),
            "min_child_samples": optuna.distributions.IntUniformDistribution(5, 100),
        },
        value=0.95,
    )
)
study.optimize(objective, n_trials=100, timeout=600)
