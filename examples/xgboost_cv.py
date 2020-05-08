"""
Optuna example that performs Cross Validation for cancer dataset
using XGBoost.
In this example, we perform cross-validation to accuracy of cancer detection
using XGBoost. We optimize both the choice of booster model and their hyper
parameters.
We have following two ways to execute this example:
(1) Execute this code directly.
    $ python xgboost_cv.py
(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize xgboost_cv.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db
"""


import optuna
from optuna.samplers import TPESampler
import os
import sklearn.datasets
import sklearn.metrics
import xgboost as xgb


# Set path to save CV results
file_path = "./tmp_results_xgbCV_optuna/"

# Create folder if it doesnt exist
if not os.path.exists(file_path):
    try:
        os.makedirs(file_path)
    except Exception:
        print("please create folder to store xgboost CV results")

# Set Constants
seed = 108
n_folds = 3

# Make the Optuna Sampler behave in a deterministically.
sampler = TPESampler(seed=seed)


# Define Optuna Objective
# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    (data, target) = sklearn.datasets.load_breast_cancer(return_X_y=True)
    dtrain = xgb.DMatrix(data, label=target)

    param = {
        "silent": 1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree"]),
        "alpha": trial.suggest_loguniform("alpha", 1e-3, 1.0),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 9)
        param["eta"] = trial.suggest_loguniform("eta", 1e-3, 1.0)
        param["gamma"] = trial.suggest_loguniform("gamma", 1e-3, 1.0)
        param["subsample"] = trial.suggest_loguniform("subsample", 0.6, 1.0)
        param["colsample_bytree"] = trial.suggest_loguniform("colsample_bytree", 0.6, 1.0)
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
        nfold=n_folds,
        stratified=True,
        early_stopping_rounds=100,
        seed=seed,
        verbose_eval=False,
    )

    # (Optional)
    # Print n_estimators in the output at each call to the objective function
    print(
        "-" * 10,
        "Trial {} has optimal trees: {}".format(trial.number, str(xgb_cv_results.shape[0])),
        "-" * 10,
    )
    print()

    # (Optional: Disable if not needed)
    # Save XGB results for Analysis; Update to your path by changing: file_path
    xgb_cv_results.to_csv(file_path + "Optuna_cv_{}.csv".format(trial.number), index=False)

    # Set n_estimators as a trial attribute; Accessible via study.trials_dataframe()
    trial.set_user_attr("n_estimators", len(xgb_cv_results))

    # Extract the best score
    best_score = xgb_cv_results["test-auc-mean"].values[-1]
    return best_score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=10)
    print()
    print(study.best_trial)
