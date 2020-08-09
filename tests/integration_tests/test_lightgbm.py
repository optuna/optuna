from functools import partial

import lightgbm as lgb
import pytest

import optuna
from optuna.integration.lightgbm import LightGBMPruningCallback
from optuna.testing.integration import create_running_trial
from optuna.testing.integration import DeterministicPruner

# If `True`, `lgb.cv(..)` will be used in the test, otherwise `lgb.train(..)` will be used.
CV_FLAGS = [False, True]


@pytest.mark.parametrize("cv", CV_FLAGS)
def test_lightgbm_pruning_callback_call(cv):
    # type: (bool) -> None

    callback_env = partial(
        lgb.callback.CallbackEnv,
        model="test",
        params={},
        begin_iteration=0,
        end_iteration=1,
        iteration=1,
    )

    if cv:
        env = callback_env(evaluation_result_list=[(("cv_agg", "binary_error", 1.0, False, 1.0))])
    else:
        env = callback_env(evaluation_result_list=[("validation", "binary_error", 1.0, False)])

    # The pruner is deactivated.
    study = optuna.create_study(pruner=DeterministicPruner(False))
    trial = create_running_trial(study, 1.0)
    pruning_callback = LightGBMPruningCallback(trial, "binary_error", valid_name="validation")
    pruning_callback(env)

    # The pruner is activated.
    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = create_running_trial(study, 1.0)
    pruning_callback = LightGBMPruningCallback(trial, "binary_error", valid_name="validation")
    with pytest.raises(optuna.TrialPruned):
        pruning_callback(env)


@pytest.mark.parametrize("cv", CV_FLAGS)
def test_lightgbm_pruning_callback(cv):
    # type: (bool) -> None

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(partial(objective, cv=cv), n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(partial(objective, cv=cv), n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0

    # Use non default validation name.
    custom_valid_name = "my_validation"
    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(lambda trial: objective(trial, valid_name=custom_valid_name, cv=cv), n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0

    # Check "maximize" direction.
    study = optuna.create_study(pruner=DeterministicPruner(True), direction="maximize")
    study.optimize(lambda trial: objective(trial, metric="auc", cv=cv), n_trials=1, catch=())
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False), direction="maximize")
    study.optimize(lambda trial: objective(trial, metric="auc", cv=cv), n_trials=1, catch=())
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0


@pytest.mark.parametrize("cv", CV_FLAGS)
def test_lightgbm_pruning_callback_errors(cv):
    # type: (bool) -> None

    # Unknown metric
    study = optuna.create_study(pruner=DeterministicPruner(False))
    with pytest.raises(ValueError):
        study.optimize(
            lambda trial: objective(trial, metric="foo_metric", cv=cv), n_trials=1, catch=()
        )

    if not cv:
        # Unknown validation name
        study = optuna.create_study(pruner=DeterministicPruner(False))
        with pytest.raises(ValueError):
            study.optimize(
                lambda trial: objective(
                    trial, valid_name="valid_1", force_default_valid_names=True
                ),
                n_trials=1,
                catch=(),
            )

    # Check consistency of study direction.
    study = optuna.create_study(pruner=DeterministicPruner(False))
    with pytest.raises(ValueError):
        study.optimize(lambda trial: objective(trial, metric="auc", cv=cv), n_trials=1, catch=())

    study = optuna.create_study(pruner=DeterministicPruner(False), direction="maximize")
    with pytest.raises(ValueError):
        study.optimize(
            lambda trial: objective(trial, metric="binary_error", cv=cv), n_trials=1, catch=()
        )


def objective(
    trial, metric="binary_error", valid_name="valid_0", force_default_valid_names=False, cv=False
):
    # type: (optuna.trial.Trial, str, str, bool, bool) -> float

    dtrain = lgb.Dataset([[1.0], [2.0], [3.0]], label=[1.0, 0.0, 1.0])
    dtest = lgb.Dataset([[1.0]], label=[1.0])

    if force_default_valid_names:
        valid_names = None
    else:
        valid_names = [valid_name]

    pruning_callback = LightGBMPruningCallback(trial, metric, valid_name=valid_name)
    if cv:
        lgb.cv(
            {"objective": "binary", "metric": ["auc", "binary_error"]},
            dtrain,
            1,
            verbose_eval=False,
            nfold=2,
            callbacks=[pruning_callback],
        )
    else:
        lgb.train(
            {"objective": "binary", "metric": ["auc", "binary_error"]},
            dtrain,
            1,
            valid_sets=[dtest],
            valid_names=valid_names,
            verbose_eval=False,
            callbacks=[pruning_callback],
        )
    return 1.0
