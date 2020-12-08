from collections import OrderedDict

import numpy as np
import pytest
import xgboost as xgb

import optuna
from optuna.integration.xgboost import use_callback_cls
from optuna.integration.xgboost import XGBoostPruningCallback
from optuna.testing.integration import create_running_trial
from optuna.testing.integration import DeterministicPruner


def test_xgboost_pruning_callback_call() -> None:
    if use_callback_cls is False:
        env = xgb.core.CallbackEnv(
            model="test",
            cvfolds=1,
            begin_iteration=0,
            end_iteration=1,
            rank=1,
            iteration=1,
            evaluation_result_list=[["validation-error", 1.0]],
        )

        # The pruner is deactivated.
        study = optuna.create_study(pruner=DeterministicPruner(False))
        trial = create_running_trial(study, 1.0)
        pruning_callback = XGBoostPruningCallback(trial, "validation-error")
        pruning_callback(env)

        # The pruner is activated.
        study = optuna.create_study(pruner=DeterministicPruner(True))
        trial = create_running_trial(study, 1.0)
        pruning_callback = XGBoostPruningCallback(trial, "validation-error")
        with pytest.raises(optuna.TrialPruned):
            pruning_callback(env)
    else:
        # The pruner is deactivated.
        study = optuna.create_study(pruner=DeterministicPruner(False))
        trial = create_running_trial(study, 1.0)
        pruning_callback = XGBoostPruningCallback(trial, "validation-logloss")
        pruning_callback.after_iteration(
            model=None, epoch=1, evals_log={"validation": OrderedDict({"logloss": [1.0]})}
        )

        # The pruner is activated.
        study = optuna.create_study(pruner=DeterministicPruner(True))
        trial = create_running_trial(study, 1.0)
        pruning_callback = XGBoostPruningCallback(trial, "validation-logloss")
        with pytest.raises(optuna.TrialPruned):
            pruning_callback.after_iteration(
                model=None, epoch=1, evals_log={"validation": OrderedDict({"logloss": [1.0]})}
            )


def test_xgboost_pruning_callback() -> None:
    if use_callback_cls:
        key = "validation-logloss"
    else:
        key = "validation-error"

    def objective(trial: optuna.trial.Trial) -> float:

        dtrain = xgb.DMatrix(np.asarray([[1.0]]), label=[1.0])
        dtest = xgb.DMatrix(np.asarray([[1.0]]), label=[1.0])

        pruning_callback = XGBoostPruningCallback(trial, key)
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
    if use_callback_cls:
        key = "test-logloss"
    else:
        key = "test-error"

    def objective(trial: optuna.trial.Trial) -> float:

        dtrain = xgb.DMatrix(np.ones((2, 1)), label=[1.0, 1.0])
        params = {
            "objective": "binary:logistic",
        }

        pruning_callback = optuna.integration.XGBoostPruningCallback(trial, key)
        xgb.cv(params, dtrain, callbacks=[pruning_callback], nfold=2)
        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0
