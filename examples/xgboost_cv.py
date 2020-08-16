"""
Optuna example that performs Cross Validation for cancer dataset using XGBoost.

In this example, we perform cross-validation to accuracy of cancer detection
using XGBoost. We optimize both the choice of booster model and their hyper
parameters.

We have following way to execute this example:

(1) Execute this code directly.
    $ python xgboost_cv.py


"""


import os
import shutil
import sklearn.datasets
import sklearn.metrics
import xgboost as xgb

import optuna


# Set constants
SEED = 108
N_FOLDS = 3
CV_RESULT_DIR = "./xgboost_cv_results"

if not os.path.exists(CV_RESULT_DIR):
    os.mkdir(CV_RESULT_DIR)


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    (data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
    dtrain = xgb.DMatrix(data, label=target)

    param = {
        "silent": 1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
        param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    xgb_cv_results = xgb.cv(
        params=param,
        dtrain=dtrain,
        num_boost_round=10000,
        nfold=N_FOLDS,
        stratified=True,
        early_stopping_rounds=100,
        seed=SEED,
        verbose_eval=False,
    )

    # Set n_estimators as a trial attribute; Accessible via study.trials_dataframe().
    trial.set_user_attr("n_estimators", len(xgb_cv_results))

    # Save cross-validation results.
    filepath = os.path.join(CV_RESULT_DIR, "{}.csv".format(trial.number))
    xgb_cv_results.to_csv(filepath, index=False)

    # Extract the best score.
    best_score = xgb_cv_results["test-auc-mean"].values[-1]
    return best_score


if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  Number of estimators: {}".format(trial.user_attrs["n_estimators"]))

    shutil.rmtree(CV_RESULT_DIR)
