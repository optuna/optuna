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

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/pruning/keras_integration.py>`__
    if you want to add a pruning callback which observes validation accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` and
            ``val_accuracy``. Please refer to `keras.Callback reference
            <https://keras.io/callbacks/#callback>`_ for further details.
        interval:
            Check if trial should be pruned every n-th epoch. By default ``interval=1`` and
            pruning is performed after every epoch. Increase ``interval`` to run several
            epochs faster before applying pruning.
     """

    def __init__(self, trial, monitor, interval=1):
        # type: (optuna.trial.Trial, str, int) -> None

        super(KerasPruningCallback, self).__init__()

        _check_keras_availability()

        self._trial = trial
        self._monitor = monitor
        self._interval = interval

    def on_epoch_end(self, epoch, logs=None):
        # type: (int, Dict[str, float]) -> None

        if (epoch + 1) % self._interval != 0:
            return

        logs = logs or {}
        current_score = logs.get(self._monitor)
        if current_score is None:
            return
        self._trial.report(float(current_score), step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)


def _check_keras_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            "Keras is not available. Please install Keras to use this feature. "
            "Keras can be installed by executing `$ pip install keras tensorflow`. "
            "For further information, please refer to the installation guide of Keras. "
            "(The actual import error is as follows: " + str(_import_error) + ")"
        )
