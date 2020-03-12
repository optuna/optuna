import optuna
from optuna import type_checking

if type_checking.TYPE_CHECKING:
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

            model.fit(X, y, callbacks=[KerasPruningCallback(trial, 'val_loss')])

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` and
            ``val_acc``. Please refer to `keras.Callback reference
            <https://keras.io/callbacks/#callback>`_ for further details.
    """

    def __init__(self, trial, monitor):
        # type: (optuna.trial.Trial, str) -> None

        super(KerasPruningCallback, self).__init__()

        _check_keras_availability()

        self._trial = trial
        self._monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        # type: (int, Dict[str, float]) -> None

        logs = logs or {}
        current_score = logs.get(self._monitor)
        if current_score is None:
            return
        self._trial.report(float(current_score), step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.exceptions.TrialPruned(message)


def _check_keras_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            "Keras is not available. Please install Keras to use this feature. "
            "Keras can be installed by executing `$ pip install keras tensorflow`. "
            "For further information, please refer to the installation guide of Keras. "
            "(The actual import error is as follows: " + str(_import_error) + ")"
        )
