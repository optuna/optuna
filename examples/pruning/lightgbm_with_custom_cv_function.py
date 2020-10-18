"""
This configurable Optuna example demonstrates the use of pruning when using
custom cross-validation functions based on LightGBM classifiers or regressors.

In this example, we optimize cross-validation accuracy of either a classification model
of breast cancer probability or a regression model of Boston housing prices,
with a custom CV function custom_cv_fun() applying LightGBM's .fit() method
to each cross-validation fold separately, instead of using the standard
built-in .cv() function.

Throughout training, an Optuna pruner observes intermediate results and if necessary
stops early unpromising trials using a callback, resulting in noticeably faster execution
times with unaffected accuracy.

The example emphasizes reproducibility, giving the user precise control over all seed
values that are being used to let her arrive at the same results over repeated studies.

You can run this example using python 3 interpreter, e.g. as follows:
    $ python lightgbm_with_custom_cv_function.py
"""

# Import packages

from datetime import datetime
from pprint import pprint

from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


# Custom functions


def custom_cv_fun(
    lgbm_params,
    X,
    y,
    objective="binary",
    eval_metric="auc",
    eval_name="valid",
    n_estimators=100,
    early_stopping_rounds=10,
    nfold=5,
    random_state=123,
    callbacks=[],
    verbose=False,
):

    # create placeholders for results
    fold_best_iterations = []
    fold_best_scores = []

    # get feature names
    feature_names = list(X.columns)

    # split data into k folds
    kf = KFold(n_splits=nfold, shuffle=True, random_state=random_state)
    splits = kf.split(X, y)

    # iterate over folds
    for train_index, valid_index in splits:

        # subset train and valid (out-of-fold) parts of the fold
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        # select estimator for the current objective
        lgbm_cv_fun = None
        if objective in ["binary", "classification"]:
            lgbm_cv_fun = LGBMClassifier
        elif objective == "regression":
            lgbm_cv_fun = LGBMRegressor

        model = lgbm_cv_fun(n_estimators=n_estimators, random_state=random_state)

        # pass hyperparameters dict to the estimator
        model.set_params(**lgbm_params)

        # train the model
        model.fit(
            X_train,
            y_train.values.ravel(),
            eval_set=(X_valid, y_valid.values.ravel()),
            eval_metric=[eval_metric],  # note: a list required
            eval_names=[eval_name],  # note: a list required
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            feature_name=feature_names,
            callbacks=callbacks,
        )

        # collect current fold data
        fold_best_iterations.append(model.best_iteration_)
        fold_best_scores.append(model.best_score_[eval_name])

    # average folds iterations numbers
    folds_data = {}
    # default to -1 iterations when early stopping is not used
    folds_data["best_iterations_mean"] = -1
    if fold_best_iterations[0] is not None:
        folds_data["best_iterations_mean"] = int(np.mean(fold_best_iterations))

    # collect metrics for best scores in each fold
    fold_best_score = {}
    for metric in fold_best_scores[0].keys():
        fold_best_score[metric] = [fold[metric] for fold in fold_best_scores]

    # avearage folds metrics (for all metrics)
    for metric in fold_best_scores[0].keys():
        folds_data["eval_mean-" + metric] = np.mean(fold_best_score[metric])

    return {
        "folds_mean_data": folds_data,
        "feature_names": feature_names,
        "fold_best_iter": fold_best_iterations,
        "fold_best_score": fold_best_score,
    }


def load_sklearn_toy_data(
    objective="binary", test_size=0.25, as_df=False, seed=123, verbose=False
):

    if objective in ["binary", "classification"]:
        # load toy classification dataset (breast cancer wisconsin)
        data, target = datasets.load_breast_cancer(return_X_y=True)

    elif objective == "regression":
        # load toy regression dataset (Boston house prices)
        data, target = datasets.load_boston(return_X_y=True)

    train_x, valid_x, train_y, valid_y = train_test_split(
        data, target, test_size=test_size, shuffle=True, random_state=seed
    )

    if verbose:
        print([i.shape for i in [train_x, valid_x, train_y, valid_y]])

    if as_df:
        # convert numpy arrays to DataFrames
        train_x_df = pd.DataFrame(train_x)
        train_x_df.columns = ["col_" + str(i) for i in train_x_df.columns]

        valid_x_df = pd.DataFrame(train_x)
        valid_x_df.columns = ["col_" + str(i) for i in valid_x_df.columns]

        train_y_df = pd.DataFrame({"y": train_y})

        valid_y_df = pd.DataFrame({"y": valid_y})

        return train_x_df, valid_x_df, train_y_df, valid_y_df

    else:
        return train_x, valid_x, train_y, valid_y


class ObjectiveCustom:
    def __init__(
        self,
        train_x_df,
        train_y_df,
        objective="binary",
        eval_metric="auc",
        eval_name="valid",
        folds=5,
        fixed_rounds=100,
        use_stopping=False,
        use_pruning=False,
        n_jobs=1,
        seed=123,
        verbosity=-1,
    ):

        self.train_x_df = train_x_df
        self.train_y_df = train_y_df
        self.objective = objective
        self.eval_metric = eval_metric
        self.eval_name = eval_name
        self.folds = folds
        self.fixed_rounds = fixed_rounds
        self.use_stopping = use_stopping
        self.use_pruning = use_pruning
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbosity = verbosity

    def __call__(self, trial):

        params = {
            "bagging_fraction": float(trial.suggest_float("bagging_fraction", 0.1, 1, log=False)),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 100, log=True),
            "feature_fraction": float(trial.suggest_float("feature_fraction", 0.1, 1, log=False)),
            "lambda_l1": float(trial.suggest_float("lambda_l1", 1e-06, 100, log=True)),
            "lambda_l2": float(trial.suggest_float("lambda_l2", 1e-06, 100, log=True)),
            "learning_rate": float(trial.suggest_float("learning_rate", 0.0001, 0.3, log=True)),
            # caution: num_boost_round should be used as argument, not params key
            # 'num_boost_round': trial.suggest_int('num_boost_round', 100, 2000, log=False),
            "num_leaves": trial.suggest_int("num_leaves", 3, 100, log=True),
            # static parameters:
            "boosting_type": "gbdt",
            "objective": self.objective,
            "metric": self.eval_metric,
            "n_jobs": self.n_jobs,
            "seed": self.seed,
            "verbose": self.verbosity,
        }

        # add a LightGBM callback for pruning
        lgbm_pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial, metric=self.eval_metric, valid_name=self.eval_name
        )

        if self.use_stopping:
            # random boosting and stopping rounds
            # will be passed as arguments
            num_boost_round = trial.suggest_int("num_boost_round", 100, 2000, log=False)
            early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 2, 500, log=True)
        else:
            num_boost_round = self.fixed_rounds
            early_stopping_rounds = None

        # train the model using custom function
        cv_results_dict = custom_cv_fun(
            lgbm_params=params,
            X=self.train_x_df,
            y=self.train_y_df,
            objective=self.objective,
            eval_metric=self.eval_metric,
            eval_name=self.eval_name,
            n_estimators=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            nfold=self.folds,
            random_state=self.seed,
            callbacks=[lgbm_pruning_callback] if self.use_pruning else [],
            verbose=self.verbosity,
        )

        # get mean CV metric from the appropriate key
        # of the dict returned by the custom CV function
        eval_mean_metric = cv_results_dict["folds_mean_data"]["eval_mean-" + self.eval_metric]

        print("Mean CV metric: %.5f" % eval_mean_metric)

        return eval_mean_metric


def main():

    # Settings

    # set number of Optuna trials
    TRIALS = 20

    # set number of CV folds
    FOLDS = 5

    # the name of the validation part of each fold,
    # passed to *both* 'eval_names' arg of .fit()
    # and to 'valid_name' arg of LightGBMPruningCallback()
    CV_EVAL_NAME = "valid"

    # whether to use early stopping
    USE_STOPPING = True

    # fix number of boosting rounds
    # note: will be overridden if USE_STOPPING is True
    FIXED_ROUNDS = -1
    if not USE_STOPPING:
        FIXED_ROUNDS = 500

    # set model objective
    OBJECTIVE = "binary"
    # OBJECTIVE = "regression"

    # set metric for model training and validation
    # as well as its optimization direction
    METRIC = "unknown"
    OPT_DIR = "unknown"
    if OBJECTIVE in ["binary", "classification"]:
        METRIC = "auc"
        OPT_DIR = "maximize"

    elif OBJECTIVE in ["regression"]:
        METRIC = "rmse"
        OPT_DIR = "minimize"

    # whether to make Optuna experiment reproducible
    # (with a repeatable set of metrics for each set of seeds)
    # at the cost of performance (single worker)
    MAKE_REPRODUCIBLE = True

    # define the number of threads
    # to be used by lightgbm
    # in each of the workers
    LGBM_NUM_THREADS = 2

    # set number of Optuna workers
    WORKERS_NUM = 2

    if MAKE_REPRODUCIBLE:
        # turn off multiprocessing in Optuna samplers to ensure reproducible results
        # (https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results)
        WORKERS_NUM = 1

    # fix seed for data partitioning (train_test_split())
    # and model training (lgbm.fit())
    SEED = 123

    # select sampler for Optuna:
    # (see more samplers at: https://optuna.readthedocs.io/en/stable/reference/samplers.html)
    # - TPE (Tree-structured Parzen Estimator) sampling algo;
    # note we fix sampler seed to make the sampling process deterministic
    SAMPLER_SEED = 123
    OPTUNA_SAMPLER = TPESampler(seed=SAMPLER_SEED)

    # whether to prune unpromising trials to speed up studies
    USE_PRUNING = True

    # select pruner for Optuna:
    # (see more pruners at: https://optuna.readthedocs.io/en/latest/reference/pruners.html)
    # - median pruner
    OPTUNA_PRUNER = MedianPruner()

    # set verbosity level for lightgbm
    VERBOSE_LGBM = -1

    # Experiment

    # Load data

    train_x_df, _, train_y_df, _ = load_sklearn_toy_data(
        objective=OBJECTIVE, test_size=0.01, as_df=True, seed=SEED
    )

    # Optimize params using Optuna

    optuna_best_metrics = {}
    optuna_best_params = {}

    # Instantiate the custom function
    objective = ObjectiveCustom(
        train_x_df,
        train_y_df,
        objective=OBJECTIVE,
        eval_metric=METRIC,
        eval_name=CV_EVAL_NAME,
        folds=FOLDS,
        fixed_rounds=FIXED_ROUNDS,
        use_stopping=USE_STOPPING,
        use_pruning=USE_PRUNING,
        n_jobs=LGBM_NUM_THREADS,
        seed=SEED,
        verbosity=VERBOSE_LGBM,
    )

    study = optuna.create_study(sampler=OPTUNA_SAMPLER, pruner=OPTUNA_PRUNER, direction=OPT_DIR)

    start_time = datetime.now()

    # run Optuna optimization over the specified number of trials
    study.optimize(objective, n_trials=TRIALS, n_jobs=WORKERS_NUM)

    optim_time_custom = datetime.now() - start_time
    print(
        "\nOptuna+custom fun. study with {:d} trials took {:.2f} s.".format(
            TRIALS, optim_time_custom.total_seconds()
        )
    )
    print("Time per trial: {:.2f} s.".format(optim_time_custom.total_seconds() / TRIALS))

    optuna_best_metrics["custom"] = study.best_value

    best_trial = study.best_trial
    optimized_best_params = best_trial.params

    # append static parameters not returned by Optuna
    static_params = {
        "boosting_type": "gbdt",
        "metric": METRIC,
        "objective": OBJECTIVE,
        "n_jobs": LGBM_NUM_THREADS,
        "seed": SEED,
        "verbosity": VERBOSE_LGBM,
    }
    all_best_params = {**optimized_best_params, **static_params}
    optuna_best_params["custom"] = all_best_params

    print(
        "\nBest mean {} for Optuna+custom fun. (reported by Optuna): {:.5f}\n".format(
            METRIC, optuna_best_metrics["custom"]
        )
    )

    print("Optuna-optimized best hyperparameters: ")
    pprint(optuna_best_params["custom"])


if __name__ == "__main__":
    main()
