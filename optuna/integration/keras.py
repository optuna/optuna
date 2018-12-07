from __future__ import absolute_import

import math

import optuna

from typing import Dict  # NOQA

try:
    from keras.callbacks import Callback
    _available = True
except ImportError as e:
    _import_error = e
    # KerasPruningExtension is disabled because Keras is not available.
    _available = False
    # This alias is required to avoid ImportError at KerasPruningExtension definition.
    Callback = object


class KerasPruningCallback(Callback):
    """Keras callback to prune unpromising trials.

    Example:

        Add a pruning callback which observes validation losses.

        .. code::

            model.fit(X, y,
                callbacks=KerasPruningCallback(trial, 'val/loss'))

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        observation_key:
            An evaluation metric for pruning, e.g., ``val/loss`` and
            ``val/acc``. Please refer to `keras.Model reference
            <https://keras.io/models/about-keras-models/>`_ for further details.
    """

    def __init__(self, trial, observation_key):
        # type: (optuna.trial.Trial, str) -> None

        _check_keras_availability()

        self.trial = trial
        self.observation_key = observation_key

    def on_epoch_end(self, epoch, logs=None):
        # type: (int, Dict[str, float]) -> None
        logs = logs or {}
        current_score = logs.get(self.observation_key)
        if current_score is None:
            return
        if math.isnan(current_score):
            return
        self.trial.report(current_score, step=epoch)
        if self.trial.should_prune(epoch):
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.structs.TrialPruned(message)


def _check_keras_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            'Keras is not available. Please install Keras to use this feature. '
            'Keras can be installed by executing `$ pip install keras tensorflow`. '
            'For further information, please refer to the installation guide of Keras. '
            '(The actual import error is as follows: ' + str(_import_error) + ')')
