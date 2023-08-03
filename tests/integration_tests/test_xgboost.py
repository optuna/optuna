import numpy as np
import pytest

import optuna
from optuna._imports import try_import
from optuna.integration.xgboost import XGBoostPruningCallback
from optuna.testing.pruners import DeterministicPruner


with try_import():
    import xgboost as xgb

pytestmark = pytest.mark.integration


def test_xgboost_pruning_callback_call() -> None:
    # The pruner is deactivated.
    study = optuna.create_study(pruner=DeterministicPruner(False))
    trial = study.ask()
    pruning_callback = XGBoostPruningCallback(trial, "validation-logloss")
    pruning_callback.after_iteration(
        model=None, epoch=1, evals_log={"validation": {"logloss": [1.0]}}
    )

    # The pruner is activated.
    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = study.ask()
    pruning_callback = XGBoostPruningCallback(trial, "validation-logloss")
    with pytest.raises(optuna.TrialPruned):
        pruning_callback.after_iteration(
            model=None, epoch=1, evals_log={"validation": {"logloss": [1.0]}}
        )


def test_xgboost_pruning_callback() -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        dtrain = xgb.DMatrix(np.asarray([[1.0]]), label=[1.0])
        dtest = xgb.DMatrix(np.asarray([[1.0]]), label=[1.0])

        pruning_callback = XGBoostPruningCallback(trial, "validation-logloss")
        xgb.train(
            {"objective": "binary:logistic"},
            dtrain,
            1,
            evals=[(dtest, "validation")],
            verbose_eval=False,
            callbacks=[pruning_callback],
        )
        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0


def test_xgboost_pruning_callback_cv() -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        dtrain = xgb.DMatrix(np.ones((2, 1)), label=[1.0, 1.0])
        params = {
            "objective": "binary:logistic",
        }

        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "test-logloss")
        xgb.cv(params, dtrain, callbacks=[pruning_callback], nfold=2)
        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0
