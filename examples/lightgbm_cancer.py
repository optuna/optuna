"""
Optuna example that optimizes a classifier configuration for cancer dataset
using LightGBM.

In this example, we optimize the validation accuracy of cancer detection
using LightGBM. We optimize both the choice of booster model and their hyper
parameters.

We have following two ways to execute this example:

(1) Execute this code directly.
    $ python lightgbm_cancer.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --storage sqlite:///example.db`
    $ optuna study optimize lightgbm_cancer.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db

"""

from __future__ import division

import lightgbm as lgb
import numpy as np
import sklearn.datasets
import sklearn.metrics
from sklearn.model_selection import train_test_split

import optuna


def objective(trial):
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)

    num_round = trial.suggest_int('num_round', 1, 500)
    param = {'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
             'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
             'num_leaves': trial.suggest_int('num_leaves', 10, 1000),
             'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0)
             }

    if param['boosting_type'] == 'dart':
        param['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
        param['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
    if param['boosting_type'] == 'goss':
        param['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
        param['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - param['top_rate'])

    bst = lgb.train(param, dtrain, num_round)
    preds = bst.predict(test_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
    return 1.0 - accuracy

if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print(study.best_trial)
