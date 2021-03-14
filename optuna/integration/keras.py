from typing import Dict
from typing import Optional
import warnings

import optuna
from optuna._deprecated import deprecated


with optuna._imports.try_import() as _imports:
    from keras.callbacks import Callback

if not _imports.is_successful():
    Callback = object  # NOQA


@deprecated(
    "2.1.0",
    text="Recent Keras release (2.4.0) simply redirects all APIs "
    "in the standalone keras package to point to tf.keras. "
    "There is now only one Keras: tf.keras. "
    "There may be some breaking changes for some workflows by upgrading to keras 2.4.0. "
    "Test before upgrading. "
    "REF:https://github.com/keras-team/keras/releases/tag/2.4.0",
)
class KerasPruningCallback(Callback):
    """Keras callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/keras/keras_integration.py>`__
    if you want to add a pruning callback which observes validation accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` and
            ``val_accuracy``. Please refer to `keras.Callback reference
            <https://keras.io/callbacks/#callback>`_ for further details.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super(KerasPruningCallback, self).__init__()

        _imports.check()

        self._trial = trial
        self._monitor = monitor

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
        trial = self._trial.storage.get_trial(self._trial._trial_id)
        if not self._trial.study.pruner.is_target_step(epoch, trial):
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
