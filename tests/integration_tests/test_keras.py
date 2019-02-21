import numpy as np

import pytest

# TODO(higumachan): remove this "try-except" section after Tensorflow supports Python 3.7.
try:
    from keras.layers import Dense
    from keras import Sequential
    _available = True
except ImportError:
    _available = False

import optuna
from optuna.integration import KerasPruningCallback
from optuna.testing.integration import DeterministicPruner


def test_keras_pruning_callback():
    # type: () -> None

    # TODO(higumachan): remove this "if" section after Tensorflow supports Python 3.7.
    if not _available:
        pytest.skip('This test requires keras '
                    'but this version can not install keras(tensorflow) with pip.')

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        model = Sequential()
        model.add(Dense(1, activation='sigmoid', input_dim=20))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(
            np.zeros((16, 20), np.float32),
            np.zeros((16, ), np.int32),
            batch_size=1,
            epochs=1,
            callbacks=[KerasPruningCallback(trial, 'acc')],
            verbose=0)

        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.structs.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.structs.TrialState.COMPLETE
    assert study.trials[0].value == 1.0


def test_keras_pruning_callback_observation_isnan():
    # type: () -> None

    # TODO(higumachan): remove this "if" section after Tensorflow supports Python 3.7.
    if not _available:
        pytest.skip('This test requires keras '
                    'but this version can not install keras(tensorflow) with pip.')

    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = study._run_trial(func=lambda _: 1.0, catch=(Exception, ))
    callback = KerasPruningCallback(trial, 'loss')

    with pytest.raises(optuna.structs.TrialPruned):
        callback.on_epoch_end(0, {'loss': 1.0})

    with pytest.raises(optuna.structs.TrialPruned):
        callback.on_epoch_end(0, {'loss': float('nan')})
