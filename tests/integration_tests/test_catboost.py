import types

import catboost as cb
import numpy as np
import pytest

import optuna
from optuna.integration.catboost import CatBoostPruningCallback
from optuna.testing.integration import create_running_trial
from optuna.testing.integration import DeterministicPruner


def test_catboost_pruning_callback_call() -> None:
    # The pruner is deactivated.
    study = optuna.create_study(pruner=DeterministicPruner(False))
    trial = create_running_trial(study, 1.0)
    pruning_callback = CatBoostPruningCallback(trial, "Logloss", "validation")
    info = types.SimpleNamespace(
        iteration=1, metrics={"learn": {"Logloss": [1.0]}, "validation": {"Logloss": [1.0]}}
    )
    assert pruning_callback.after_iteration(info) is True

    # The pruner is activated.
    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = create_running_trial(study, 1.0)
    pruning_callback = CatBoostPruningCallback(trial, "Logloss", "validation")
    info = types.SimpleNamespace(
        iteration=1, metrics={"learn": {"Logloss": [1.0]}, "validation": {"Logloss": [1.0]}}
    )
    assert pruning_callback.after_iteration(info) is False


def test_catboost_pruning_callback() -> None:
    def objective(trial: optuna.trial.Trial) -> float:

        train_x = np.asarray([[1.0], [2.0]])
        train_y = np.asarray([[1.0], [0.0]])
        valid_x = np.asarray([[1.0], [2.0]])
        valid_y = np.asarray([[1.0], [0.0]])

        pruning_callback = CatBoostPruningCallback(trial, "AUC")
        param = {
            "objective": "Logloss",
            "eval_metric": "AUC",
        }

        gbm = cb.CatBoostClassifier(**param)
        gbm.fit(
            train_x,
            train_y,
            eval_set=[(valid_x, valid_y)],
            verbose=0,
            callbacks=[pruning_callback],
        )

        # evoke pruning manually.
        pruning_callback.check_pruned()

        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0


METRICS = ["AUC", "Accuracy"]
VALID_NAMES = ["validation_0", "validation_1"]


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("valid_name", VALID_NAMES)
def test_catboost_pruning_callback_init_param(metric: str, valid_name: str) -> None:
    def objective(trial: optuna.trial.Trial) -> float:

        train_x = np.asarray([[1.0], [2.0]])
        train_y = np.asarray([[1.0], [0.0]])
        valid_x = np.asarray([[1.0], [2.0]])
        valid_y = np.asarray([[1.0], [0.0]])

        pruning_callback = CatBoostPruningCallback(trial, metric, valid_name)
        param = {
            "objective": "Logloss",
            "eval_metric": metric,
        }

        gbm = cb.CatBoostClassifier(**param)
        gbm.fit(
            train_x,
            train_y,
            eval_set=[(valid_x, valid_y), (valid_x, valid_y)],
            verbose=0,
            callbacks=[pruning_callback],
        )

        # evoke pruning manually.
        pruning_callback.check_pruned()

        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0


@pytest.mark.parametrize(
    "metric, valid_name",
    [
        ("foo_metric", "validation"),
        ("AUC", "foo_valid"),
    ],
)
def test_catboost_pruning_callback_errors(metric: str, valid_name: str) -> None:
    def objective(trial: optuna.trial.Trial) -> float:

        train_x = np.asarray([[1.0], [2.0]])
        train_y = np.asarray([[1.0], [0.0]])
        valid_x = np.asarray([[1.0], [2.0]])
        valid_y = np.asarray([[1.0], [0.0]])

        pruning_callback = CatBoostPruningCallback(trial, metric, valid_name)
        param = {
            "objective": "Logloss",
            "eval_metric": "AUC",
        }

        gbm = cb.CatBoostClassifier(**param)
        gbm.fit(
            train_x,
            train_y,
            eval_set=[(valid_x, valid_y)],
            verbose=0,
            callbacks=[pruning_callback],
        )

        # evoke pruning manually.
        pruning_callback.check_pruned()

        return 1.0

    # Unknown validation name or metric
    study = optuna.create_study(pruner=DeterministicPruner(False))
    # catboost terminates with a SystemError.
    with pytest.raises(SystemError):
        study.optimize(objective, n_trials=1)
