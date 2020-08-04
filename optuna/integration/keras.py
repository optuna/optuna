import optuna
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Dict  # NOQA

with optuna._imports.try_import() as _imports:
    from keras.callbacks import Callback

if not _imports.is_successful():
    Callback = object  # NOQA


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

        _imports.check()

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
