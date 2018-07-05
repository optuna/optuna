"""
PFNOpt example that optimizes a classifier configuration for cancer dataset
using XGBoost.

In this example, we optimize the validation accuracy of cancer detection
using XGBoost. We optimize both the choice of booster model and their hyper
parameters.

We have following two ways to execute this example:

(1) Execute this code directly.
    $ python xgboost_cancer.py

(2) Execute throurgh CLI.
    $ pfnopt minimize xgboost_cancer.py objective --create-study --n-trials=100

"""

from __future__ import division

import sklearn.datasets
import sklearn.metrics
import xgboost as xgb


def objective(trial):
    (data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)

    train_size = int(data.shape[0] * 0.75)

    train_x = data[:train_size, :]
    train_y = target[:train_size]
    test_x = data[train_size:, :]
    test_y = target[train_size:]

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)

    booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart'])

    n_round = int(trial.suggest_uniform('n_round', 1, 10))
    param = {'silent': 1, 'objective': 'binary:logistic'}

    param['lambda'] = trial.suggest_loguniform('lambda', 1e-8, 1.0)
    param['alpha'] = trial.suggest_loguniform('alpha', 1e-8, 1.0)
    if booster == 'gbtree' or booster == 'dart':
        param['max_depth'] = int(trial.suggest_uniform('max_depth', 1, 10))
        param['ets'] = trial.suggest_loguniform('eta', 1e-8, 1.0)
        param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
        param['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    if booster == 'dart':
        param['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        param['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
        param['rate_drop'] = trial.suggest_loguniform('rate_drop', 1e-8, 1.0)
        param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)

    bst = xgb.train(param, dtrain, n_round)
    preds = bst.predict(dtest)
    pred_labels = [round(value) for value in preds]
    accuracy = sklearn.metrics.accuracy_score(dtest.get_label(), pred_labels)
    return 1.0 - accuracy


if __name__ == '__main__':
    import pfnopt
    study = pfnopt.minimize(objective, n_trials=100)
    print(study.best_trial)
