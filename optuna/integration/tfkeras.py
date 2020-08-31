from typing import Any
from typing import Dict
from typing import Optional

import optuna

with optuna._imports.try_import() as _imports:
    from tensorflow.keras.callbacks import Callback

if not _imports.is_successful():
    Callback = object  # NOQA


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

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:

        super(TFKerasPruningCallback, self).__init__()

        _imports.check()

        self._trial = trial
        self._monitor = monitor

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:

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
