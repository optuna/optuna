import pytest
import xgboost as xgb

import optuna
from optuna.integration.xgboost import XGBoostPruningCallback
from optuna.testing.integration import create_running_trial
from optuna.testing.integration import DeterministicPruner


def test_xgboost_pruning_callback_call():
    # type: () -> None

    env = xgb.core.CallbackEnv(
        model='test',
        cvfolds=1,
        begin_iteration=0,
        end_iteration=1,
        rank=1,
        iteration=1,
        evaluation_result_list=[['validation-error', 1.]])

    # The pruner is deactivated.
    study = optuna.create_study(pruner=DeterministicPruner(False))
    trial = create_running_trial(study, 1.0)
    pruning_callback = XGBoostPruningCallback(trial, 'validation-error')
    pruning_callback(env)

    # The pruner is activated.
    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = create_running_trial(study, 1.0)
    pruning_callback = XGBoostPruningCallback(trial, 'validation-error')
    with pytest.raises(optuna.structs.TrialPruned):
        pruning_callback(env)


def test_xgboost_pruning_callback():
    # type: () -> None

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        dtrain = xgb.DMatrix([[1.]], label=[1.])
        dtest = xgb.DMatrix([[1.]], label=[1.])

        pruning_callback = XGBoostPruningCallback(trial, 'validation-error')
        xgb.train({
            'silent': 1,
            'objective': 'binary:logistic'
        },
            dtrain,
            1,
            evals=[(dtest, 'validation')],
            verbose_eval=False,
            callbacks=[pruning_callback])
        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.structs.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.structs.TrialState.COMPLETE
    assert study.trials[0].value == 1.
