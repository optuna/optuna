"""

plot_terminator_improvement
===========================

.. autofunction:: optuna.visualization.matplotlib.plot_terminator_improvement

The following code snippet shows how to plot improvement potentials,
together with cross-validation errors.

"""

from lightgbm import LGBMClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import optuna
from optuna.terminator import report_cross_validation_scores
from optuna.visualization.matplotlib import plot_terminator_improvement


def objective(trial):
    X, y = load_wine(return_X_y=True)
    clf = LGBMClassifier(
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        num_leaves=trial.suggest_int("num_leaves", 2, 256),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
        subsample=trial.suggest_float("subsample", 0.4, 1.0),
        subsample_freq=trial.suggest_int("subsample_freq", 1, 7),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
    )
    scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True))
    report_cross_validation_scores(trial, scores)
    return scores.mean()


study = optuna.create_study()
study.optimize(objective, n_trials=30)

plot_terminator_improvement(study, plot_error=True)
