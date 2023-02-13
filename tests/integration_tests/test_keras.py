import numpy as np
import pytest

import optuna
from optuna._imports import try_import
from optuna.integration import KerasPruningCallback
from optuna.testing.pruners import DeterministicPruner


with try_import():
    from keras import Sequential
    from keras.layers import Dense

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("interval, epochs", [(1, 1), (2, 1), (2, 2)])
def test_keras_pruning_callback(interval: int, epochs: int) -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        model = Sequential()
        model.add(Dense(1, activation="sigmoid", input_dim=20))
        model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(
            np.zeros((16, 20), np.float32),
            np.zeros((16,), np.int32),
            batch_size=1,
            epochs=epochs,
            callbacks=[KerasPruningCallback(trial, "accuracy", interval=interval)],
            verbose=0,
        )

        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    if interval <= epochs:
        assert study.trials[0].state == optuna.trial.TrialState.PRUNED
    else:
        assert study.trials[0].state == optuna.trial.TrialState.COMPLETE

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0


def test_keras_pruning_callback_observation_isnan() -> None:
    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = study.ask()
    callback = KerasPruningCallback(trial, "loss")

    with pytest.raises(optuna.TrialPruned):
        callback.on_epoch_end(0, {"loss": 1.0})

    with pytest.raises(optuna.TrialPruned):
        callback.on_epoch_end(0, {"loss": float("nan")})


def test_keras_pruning_callback_monitor_is_invalid() -> None:
    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = study.ask()
    callback = KerasPruningCallback(trial, "InvalidMonitor")

    with pytest.warns(UserWarning):
        callback.on_epoch_end(0, {"loss": 1.0})
