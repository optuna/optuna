from typing import Any

import optuna


with optuna._imports.try_import() as _imports:
    from skorch.callbacks import Callback
    from skorch.net import NeuralNet

if not _imports.is_successful():
    Callback = object  # NOQA


class SkorchPruningCallback(Callback):
    """Skorch callback to prune unpromising trials.

    .. versionadded:: 2.1.0

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g. ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries,
            i.e., ``net.histroy``. The names thus depend on how this dictionary
            is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:

        _imports.check()

        super().__init__()
        self._trial = trial
        self._monitor = monitor

    def on_epoch_end(self, net: "NeuralNet", **kwargs: Any) -> None:
        history = net.history
        if not history:
            return
        epoch = len(history) - 1
        current_score = history[-1, self._monitor]
        self._trial.report(current_score, epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
