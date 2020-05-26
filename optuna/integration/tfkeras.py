import optuna
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Dict  # NOQA

try:
    from tensorflow.keras.callbacks import Callback

    _available = True
except ImportError as e:
    _import_error = e
    # TFKerasPruningCallback is disabled because TensorFlow is not available.
    _available = False
    Callback = object


class TFKerasPruningCallback(Callback):
    """tf.keras callback to prune unpromising trials.

    This callback is intend to be compatible for TensorFlow v1 and v2,
    but only tested with TensorFlow v1.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/pruning/tfkeras_integration.py>`__
    if you want to add a pruning callback which observes the validation accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or ``val_acc``.
    """

    def __init__(self, trial, monitor):
        # type: (optuna.trial.Trial, str) -> None

        super(TFKerasPruningCallback, self).__init__()

        _check_tensorflow_availability()

        self._trial = trial
        self._monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        # type: (int, Dict[str, Any]) -> None

        logs = logs or {}
        current_score = logs.get(self._monitor)

        if current_score is None:
            return

        # Report current score and epoch to Optuna's trial.
        self._trial.report(float(current_score), step=epoch)

        # Prune trial if needed
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)


def _check_tensorflow_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            "TensorFlow is not available. Please install TensorFlow to use this feature. "
            "TensorFlow can be installed by executing `$ pip install tensorflow`. "
            "For further information, please refer to the installation guide of TensorFlow. "
            "(The actual import error is as follows: " + str(_import_error) + ")"
        )
