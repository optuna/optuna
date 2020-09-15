from typing import Any

import optuna
from optuna._experimental import experimental
from optuna._imports import try_import


with try_import() as _imports:
    from catalyst.dl import Callback

if not _imports.is_successful():
    Callback = object  # NOQA


@experimental("2.0.0")
class CatalystPruningCallback(Callback):
    """Catalyst callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/catalyst_simple.py>`_ if you want to add a pruning callback
    which observes the accuracy of Catalyst's ``SupervisedRunner``.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        metric (str):
            Name of a metric, which is passed to `catalyst.core.State.valid_metrics` dictionary to
            fetch the value of metric computed on validation set. Pruning decision is made based
            on this value.
    """

    def __init__(self, trial: optuna.trial.Trial, metric: str = "loss") -> None:

        # set order=1000 to run pruning callback after other callbacks
        # refer to `catalyst.core.CallbackOrder`
        _imports.check()
        super(CatalystPruningCallback, self).__init__(order=1000)

        self._trial = trial
        self.metric = metric

    def on_epoch_end(self, state: Any) -> None:

        current_score = state.valid_metrics[self.metric]
        self._trial.report(current_score, state.epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(state.epoch)
            raise optuna.TrialPruned(message)
