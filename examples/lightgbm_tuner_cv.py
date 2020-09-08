"""
Optuna example that optimizes a classifier configuration for cancer dataset using LightGBM tuner.

In this example, we optimize the cross-validated log loss of cancer detection.

"""
import sklearn.datasets
from sklearn.model_selection import KFold

import optuna.integration.lightgbm as lgb

if __name__ == "__main__":
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    dtrain = lgb.Dataset(data, label=target)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }

    tuner = lgb.LightGBMTunerCV(
        params, dtrain, verbose_eval=100, early_stopping_rounds=100, folds=KFold(n_splits=3)
    )

    tuner.run()

    print("Best score:", tuner.best_score)
    best_params = tuner.best_params
    print("Best params:", best_params)
    print("  Params: ")
    for key, value in best_params.items():
        print("    {}: {}".format(key, value))
