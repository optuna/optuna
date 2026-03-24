"""
Hyperparameters optimization for time series with Optuna and TimeSeriesSplit
============================================================================

Optuna is a powerful tool to optimize hyperparameters through cross-validation.
However, when dealing with time-series, cross-validation splits must
retain the data order to avoid data leakages and biased results.

This is a subtle issue: if temporal order is not retained during cv,
hyperparameters will be optimized and the model tuned. No error will
be thrown, as it does not depend on the cross-validation strategy's correctness.
Still, outcomes will be unreliable and biased.

This tutorial showcases how to combine Optuna hyperparameter optimization with
scikit-learn's TimeSeriesSplit for a proper time-series validation.
"""

###################################################################################################
# Why Standard validation fails for time-series?
#
# By nature, time series data is time dependent, i.e it follows a chronological order that validation
# must account for.
# Since Standard cross-validation randomly shuffles data while splitting it into k folds, it
# breaks time-series temporal dependencies. This issue is subtle (as hyperparameters will still be optimized with no errors),
# but it leads to future-to-past data leakage and overly optimistic results.
# Time-series cross validation (TimeSeriesSplit) implements a solution. It retains the temporal order by
# forcing training data to always precede validation data (usually through an expanding window mechanism).
# This prevents data leakage and allows for proper and unbiased performance evaluation.
#
# Note. To ensure proper time series processing, dependency structure must be kept
# during the train-validation-split too. However, this is outside the tutorial scope.


###################################################################################################
# IMPORT NEEDED PACKAGES

# data manipulation
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit

# xgboosting
from xgboost import XGBRegressor

# metrics
from sklearn.metrics import mean_squared_error

# set seed for reproducibility
np.random.seed(42)

###################################################################################################
# DATASET LOADING AND PREPROCESSING
# The dataset is sklearn diabetes dataset. Even though it is not a time-series dataset,
# we simulate a time-series structure by imposing an artificial temporal order and creating lag features


# data loading & pre-processing function
def data_loading_and_processing():
    # import dataset package
    from sklearn.datasets import load_diabetes

    # Load dataset
    data, target = load_diabetes(return_X_y=True, as_frame=True)
    # Combine into a single DataFrame
    df = data.copy()
    # define target
    df["target"] = target
    # Create a pseudo time index
    df["date"] = pd.date_range(start="2000-01-01", periods=len(df), freq="D")
    df.set_index("date", inplace=True)
    # Sort by date
    df.sort_index(inplace=True)
    # Create lag features on the target
    df["target_lag1"] = df["target"].shift(1)
    df["target_lag2"] = df["target"].shift(2)
    df["target_lag3"] = df["target"].shift(3)
    # create lag feature of bmi (more complexity)
    df["bmi_lag1"] = df["bmi"].shift(1)
    # Drop missing values due to lagging
    df.dropna(inplace=True)
    # define the target
    y = df["target"]
    # keep only regressors
    X = df.drop(columns=["target"])
    # return df
    return X, y


# build the regressor df and the target
X, y = data_loading_and_processing()


###################################################################################################
# TRAIN-VALIDATION SPLIT
# With time series, train-validation split is done taking into consideration time dependencies.
# A split % is defined, then data are split keeping their temporal order.

# define the train-val split %
split_index = int(len(X) * 0.85)
# split X
x_train, x_val = X.iloc[:split_index], X.iloc[split_index:]
# split y
y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]

###################################################################################################
# TIME SERIES CROSS-VALIDATION
# Define number of cv splits and define train/test indices to split time-ordered data
# through TimeSeriesSplit

# define cv number
cv_splits = 5
# define time-series cross validation
tscv = TimeSeriesSplit(n_splits=cv_splits)

###################################################################################################
# HYPERPARAMETER OPTIMIZATION WITH OPTUNA
# Set up an objective that:
# - suggests params starting points
# - defines the model XGBoost
# - splits cv based on tscv, predicts rmse score, appends them to score list
# - return the average rmse score
# Then create an Optuna study to optimize the model's performance.
# At the end, show and store best parameters along with best model's outcomes.
# Note. create_study has direction = "minimize" as the smaller the rmse, the more precise is the model.


# define an  Optuna objective
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 3),
        "gamma": trial.suggest_float("gamma", 0, 1, step=0.5),
    }
    # perform 5 fold time-series cross validation
    scores = []
    # time-series split with TimeSeriesSplit
    for train_idx, valid_idx in tscv.split(x_train):
        X_tr, X_val = x_train.iloc[train_idx], x_train.iloc[valid_idx]
        y_tr, y_val_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]
        # define the model
        model = XGBRegressor(**params, eval_metric="rmse", random_state=42, verbosity=0)
        # fit the model
        model.fit(X_tr, y_tr)
        # predict
        preds = model.predict(X_val)
        # compute rmse
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        # add rmse to score list
        scores.append(rmse)
    # return cv avg score
    return np.mean(scores)


# create a study object
study = optuna.create_study(direction="minimize")
# optimize the objective function
study.optimize(objective, n_trials=20)

# show best trial rmse
print(f"Best trial RMSE: {study.best_trial.value}")
# show best params
print(f"Best hyperparameters: {study.best_trial.params}")
# store best params
best_params = study.best_trial.params

###################################################################################################
# FINAL RE-TRAINING & ASSESSMENT WITH TOP-PERFORMING PARAMETERS
# train the XGBoost with the top-performing parameters on the entire training dataset.

# define the tuned model
xgb_tuned = XGBRegressor(
    **best_params,  # top performing params found by Optuna
    eval_metric="rmse",
    random_state=42,
    verbosity=0,
)
# re-fit the top-model on the entire training df
xgb_tuned.fit(x_train, y_train)
# predict
preds = xgb_tuned.predict(x_val)
# compute tuned model rmse
rmse = np.sqrt(mean_squared_error(y_val, preds))
# print it
print("Tuned model RMSE:")
print(rmse)
