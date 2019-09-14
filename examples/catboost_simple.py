"""
Optuna example that optimizes a classifier configuration for cancer dataset using
Catboost.

In this example, we optimize the validation accuracy of cancer detection using
Catboost. We optimize both the choice of booster model and their hyperparameters.

We have following two ways to execute this example:

(1) Execute this code directly.
    $ python catboost_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize catboost_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db

"""

import catboost as cb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import optuna


def objective(trial):
    data, target = load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.3)

    param = {
        'objective': trial.suggest_categorical('objective', ['Logloss', 'CrossEntropy']),
        'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.01, 0.1),
        'depth': trial.suggest_int('depth', 1, 12),
        'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type',
                                                    ['Bayesian', 'Bernoulli', 'MVS']),
        'used_ram_limit': '3gb'
    }

    if param['bootstrap_type'] == 'Bayesian':
        param['bagging_temperature'] = trial.suggest_uniform('bagging_temperature', 0, 10)
    elif param['bootstrap_type'] == 'Bernoulli':
        param['subsample'] = trial.suggest_uniform('subsample', 0, 1)

    gbm = cb.CatBoostClassifier(**param)

    gbm.fit(train_x, train_y, eval_set=[(test_x, test_y)], verbose=0, early_stopping_rounds=100)

    preds = gbm.predict(test_x)
    pred_labels = np.rint(preds)
    accuracy = accuracy_score(test_y, pred_labels)
    return accuracy


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, timeout=600)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
