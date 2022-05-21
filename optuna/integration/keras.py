from typing import Dict
from typing import Optional
import warnings

import optuna
from optuna._deprecated import deprecated_class


with optuna._imports.try_import() as _imports:
    from keras.callbacks import Callback

if not _imports.is_successful():
    Callback = object  # NOQA

_keras_pruning_callback_deprecated_msg = (
    "Recent Keras release (2.4.0) simply redirects "
    "all APIs in the standalone keras package to point to tf.keras. "
    "There is now only one Keras: tf.keras. "
    "There may be some breaking changes for some workflows by upgrading to keras 2.4.0. "
    "Test before upgrading. "
    "REF: https://github.com/keras-team/keras/releases/tag/2.4.0. "
    "There is an alternative callback function that can be used instead: "
    ":class:`~optuna.integration.TFKerasPruningCallback`"
)


@deprecated_class("2.1.0", "4.0.0", text=_keras_pruning_callback_deprecated_msg)
class KerasPruningCallback(Callback):
    """Keras callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    keras/keras_integration.py>`__
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

    def __init__(self, trial: optuna.trial.Trial, monitor: str, interval: int = 1) -> None:
        super().__init__()

        _imports.check()

        self._trial = trial
        self._monitor = monitor
        self._interval = interval

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        if (epoch + 1) % self._interval != 0:
            return

        logs = logs or {}
        current_score = logs.get(self._monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self._monitor)
            )
            warnings.warn(message)
            return
        self._trial.report(float(current_score), step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
