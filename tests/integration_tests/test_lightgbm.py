import lightgbm as lgb
import pytest

import optuna
from optuna.integration.lightgbm import LightGBMPruningCallback
from optuna.testing.integration import DeterministicPruner


def test_lightgbm_pruning_callback_call():
    # type: () -> None

    env = lgb.callback.CallbackEnv(
        model='test',
        params={},
        begin_iteration=0,
        end_iteration=1,
        iteration=1,
        evaluation_result_list=[('validation', 'binary_error', 1., False)])

    # The pruner is deactivated.
    study = optuna.create_study(pruner=DeterministicPruner(False))
    trial = study._run_trial(func=lambda _: 1.0, catch=(Exception,))
    pruning_callback = LightGBMPruningCallback(trial, 'binary_error', valid_name='validation')
    pruning_callback(env)

    # The pruner is activated.
    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = study._run_trial(func=lambda _: 1.0, catch=(Exception,))
    pruning_callback = LightGBMPruningCallback(trial, 'binary_error', valid_name='validation')
    with pytest.raises(optuna.structs.TrialPruned):
        pruning_callback(env)


def test_lightgbm_pruning_callback():
    # type: () -> None

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        dtrain = lgb.Dataset([[1.]], label=[1.])
        dtest = lgb.Dataset([[1.]], label=[1.])

        pruning_callback = LightGBMPruningCallback(trial, 'binary_error')
        lgb.train({'objective': 'binary', 'metric': 'binary_error'}, dtrain, 1,
                  valid_sets=[dtest], verbose_eval=False, callbacks=[pruning_callback])
        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.structs.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.structs.TrialState.COMPLETE
    assert study.trials[0].value == 1.


def test_lightgbm_pruning_callback_with_custom_validation_name():
    # type: () -> None

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        dtrain = lgb.Dataset([[1.]], label=[1.])
        dtest = lgb.Dataset([[1.]], label=[1.])

        custom_valid_name = 'my_validation'
        pruning_callback = LightGBMPruningCallback(trial, 'binary_error',
                                                   valid_name=custom_valid_name)
        lgb.train({'objective': 'binary', 'metric': 'binary_error'}, dtrain, 1,
                  valid_sets=[dtest], valid_names=[custom_valid_name],
                  verbose_eval=False, callbacks=[pruning_callback])
        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.structs.TrialState.COMPLETE
    assert study.trials[0].value == 1.


def test_lightgbm_pruning_callback_errors():
    # type: () -> None

    def objective(trial, observation_key, valid_name='valid_0'):
        # type: (optuna.trial.Trial, str, str) -> float

        dtrain = lgb.Dataset([[1.]], label=[1.])
        dtest = lgb.Dataset([[1.]], label=[1.])

        pruning_callback = LightGBMPruningCallback(trial, observation_key, valid_name=valid_name)
        lgb.train({'objective': 'binary', 'metric': ['auc', 'binary_error']}, dtrain, 1,
                  valid_sets=[dtest], verbose_eval=False, callbacks=[pruning_callback])
        return 1.0

    # "maximize" direction isn't supported yet.
    study = optuna.create_study(pruner=DeterministicPruner(False))
    trial = study._run_trial(lambda trial: objective(trial, 'auc'), catch=(ValueError,))
    frozen_trial = study.storage.get_trial(trial.trial_id)

    expected_message_prefix = "Setting trial status as TrialState.FAIL because of the following " \
                              "error: ValueError('Pruning using me"
    assert frozen_trial.state == optuna.structs.TrialState.FAIL
    assert frozen_trial.system_attrs['fail_reason'][:100] == expected_message_prefix

    # Unknown observation key (i.e., metric name)
    study = optuna.create_study(pruner=DeterministicPruner(False))
    trial = study._run_trial(lambda trial: objective(trial, 'foo_metric'), catch=(ValueError,))
    frozen_trial = study.storage.get_trial(trial.trial_id)

    expected_message_prefix = "Setting trial status as TrialState.FAIL because of the following " \
                              "error: ValueError('The entry associ"
    assert frozen_trial.state == optuna.structs.TrialState.FAIL
    assert frozen_trial.system_attrs['fail_reason'][:100] == expected_message_prefix

    # Unknown validation name
    study = optuna.create_study(pruner=DeterministicPruner(False))
    trial = study._run_trial(lambda trial: objective(trial, 'binary_error', 'valid_1'),
                             catch=(ValueError,))
    frozen_trial = study.storage.get_trial(trial.trial_id)

    expected_message_prefix = "Setting trial status as TrialState.FAIL because of the following " \
                              "error: ValueError('The entry associ"
    assert frozen_trial.state == optuna.structs.TrialState.FAIL
    assert frozen_trial.system_attrs['fail_reason'][:100] == expected_message_prefix
