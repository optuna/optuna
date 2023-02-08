from functools import partial
from unittest.mock import patch

import numpy as np
import pytest

import optuna
from optuna._imports import try_import
from optuna.integration.lightgbm import LightGBMPruningCallback
from optuna.testing.pruners import DeterministicPruner


with try_import():
    import lightgbm as lgb

pytestmark = pytest.mark.integration

# If `True`, `lgb.cv(..)` will be used in the test, otherwise `lgb.train(..)` will be used.
CV_FLAGS = [False, True]


@pytest.mark.parametrize("cv", CV_FLAGS)
def test_lightgbm_pruning_callback_call(cv: bool) -> None:
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
    trial = study.ask()
    pruning_callback = LightGBMPruningCallback(trial, "binary_error", valid_name="validation")
    pruning_callback(env)

    # The pruner is activated.
    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = study.ask()
    pruning_callback = LightGBMPruningCallback(trial, "binary_error", valid_name="validation")
    with pytest.raises(optuna.TrialPruned):
        pruning_callback(env)


@pytest.mark.parametrize("cv", CV_FLAGS)
def test_lightgbm_pruning_callback(cv: bool) -> None:
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


@pytest.mark.parametrize(
    "cv, interval, num_boost_round",
    [
        (True, 1, 1),
        (True, 2, 1),
        (True, 2, 2),
        (False, 1, 1),
        (False, 2, 1),
        (False, 2, 2),
    ],
)
def test_lightgbm_pruning_callback_with_interval(
    cv: bool, interval: int, num_boost_round: int
) -> None:
    study = optuna.create_study(pruner=DeterministicPruner(False))

    with patch("optuna.trial.Trial.report") as mock:
        study.optimize(
            partial(objective, cv=cv, interval=interval, num_boost_round=num_boost_round),
            n_trials=1,
        )

        if interval <= num_boost_round:
            assert mock.call_count == 1
        else:
            assert mock.call_count == 0

        assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
        assert study.trials[0].value == 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(
        partial(objective, cv=cv, interval=interval, num_boost_round=num_boost_round), n_trials=1
    )
    if interval > num_boost_round:
        assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    else:
        assert study.trials[0].state == optuna.trial.TrialState.PRUNED


@pytest.mark.parametrize("cv", CV_FLAGS)
def test_lightgbm_pruning_callback_errors(cv: bool) -> None:
    # Unknown metric.
    study = optuna.create_study(pruner=DeterministicPruner(False))
    with pytest.raises(ValueError):
        study.optimize(
            lambda trial: objective(trial, metric="foo_metric", cv=cv), n_trials=1, catch=()
        )

    if not cv:
        # Unknown validation name.
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
    trial: optuna.trial.Trial,
    metric: str = "binary_error",
    valid_name: str = "valid_0",
    interval: int = 1,
    num_boost_round: int = 1,
    force_default_valid_names: bool = False,
    cv: bool = False,
) -> float:
    dtrain = lgb.Dataset(np.asarray([[1.0], [2.0], [3.0], [4.0]]), label=[1.0, 0.0, 1.0, 0.0])
    dtest = lgb.Dataset(np.asarray([[1.0]]), label=[1.0])

    if force_default_valid_names:
        valid_names = None
    else:
        valid_names = [valid_name]

    verbose_callback = lgb.log_evaluation()
    pruning_callback = LightGBMPruningCallback(
        trial, metric, valid_name=valid_name, report_interval=interval
    )
    if cv:
        lgb.cv(
            {"objective": "binary", "metric": ["auc", "binary_error"]},
            dtrain,
            num_boost_round,
            nfold=2,
            callbacks=[verbose_callback, pruning_callback],
        )
    else:
        lgb.train(
            {"objective": "binary", "metric": ["auc", "binary_error"]},
            dtrain,
            num_boost_round,
            valid_sets=[dtest],
            valid_names=valid_names,
            callbacks=[verbose_callback, pruning_callback],
        )
    return 1.0
