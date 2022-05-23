import types

import catboost as cb
import numpy as np
import pytest

import optuna
from optuna.integration.catboost import CatBoostPruningCallback
from optuna.testing.integration import DeterministicPruner


def test_catboost_pruning_callback_call() -> None:
    # The pruner is deactivated.
    study = optuna.create_study(pruner=DeterministicPruner(False))
    trial = study.ask()
    pruning_callback = CatBoostPruningCallback(trial, "Logloss")
    info = types.SimpleNamespace(
        iteration=1, metrics={"learn": {"Logloss": [1.0]}, "validation": {"Logloss": [1.0]}}
    )
    assert pruning_callback.after_iteration(info)

    # The pruner is activated.
    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = study.ask()
    pruning_callback = CatBoostPruningCallback(trial, "Logloss")
    info = types.SimpleNamespace(
        iteration=1, metrics={"learn": {"Logloss": [1.0]}, "validation": {"Logloss": [1.0]}}
    )
    assert not pruning_callback.after_iteration(info)


METRICS = ["AUC", "Accuracy"]
EVAL_SET_INDEXES = [None, 0, 1]


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("eval_set_index", EVAL_SET_INDEXES)
def test_catboost_pruning_callback_init_param(metric: str, eval_set_index: int) -> None:
    def objective(trial: optuna.trial.Trial) -> float:

        train_x = np.asarray([[1.0], [2.0]])
        train_y = np.asarray([[1.0], [0.0]])
        valid_x = np.asarray([[1.0], [2.0]])
        valid_y = np.asarray([[1.0], [0.0]])

        if eval_set_index is None:
            eval_set = [(valid_x, valid_y)]
            pruning_callback = CatBoostPruningCallback(trial, metric)
        else:
            eval_set = [(valid_x, valid_y), (valid_x, valid_y)]
            pruning_callback = CatBoostPruningCallback(trial, metric, eval_set_index)

        param = {
            "objective": "Logloss",
            "eval_metric": metric,
        }

        gbm = cb.CatBoostClassifier(**param)
        gbm.fit(
            train_x,
            train_y,
            eval_set=eval_set,
            verbose=0,
            callbacks=[pruning_callback],
        )

        # Invoke pruning manually.
        pruning_callback.check_pruned()

        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0


@pytest.mark.skip(reason="Temporally skip due to unknown CatBoost error.")
@pytest.mark.parametrize(
    "metric, eval_set_index",
    [
        ("foo_metric", None),
        ("AUC", 100),
    ],
)
def test_catboost_pruning_callback_errors(metric: str, eval_set_index: int) -> None:
    def objective(trial: optuna.trial.Trial) -> float:

        train_x = np.asarray([[1.0], [2.0]])
        train_y = np.asarray([[1.0], [0.0]])
        valid_x = np.asarray([[1.0], [2.0]])
        valid_y = np.asarray([[1.0], [0.0]])

        pruning_callback = CatBoostPruningCallback(trial, metric, eval_set_index)
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

        # Invoke pruning manually.
        pruning_callback.check_pruned()

        return 1.0

    # Unknown validation name or metric.
    study = optuna.create_study(pruner=DeterministicPruner(False))
    # Catboost terminates with a SystemError.
    with pytest.raises(SystemError):
        study.optimize(objective, n_trials=1)
