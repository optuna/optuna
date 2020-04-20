import optuna


try:
    from catalyst.dl import Callback

    _available = True
except ImportError as e:
    _import_error = e
    # CatalystPruningCallback is disabled because Catalyst is not available.
    _available = False
    Callback = object


class CatalystPruningCallback(Callback):
    """Catalyst callback to prune unpromising trials.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        metric (str):
            Name of a metric, which is passed to `catalyst.core.State.valid_metrics` dictionary to fetch
            the value of metric computed on validation set. Pruning decision is made based on this value.
    """

    def __init__(self, trial, metric="loss"):
        # type: (optuna.trial.Trial, str) -> None

        # set order=1000 to run pruning callback after other callbacks (ref `catalyst.core.CallbackOrder`)
        super(CatalystPruningCallback, self).__init__(order=1000)
        _check_catalyst_availability()

        self._trial = trial
        self.metric = metric

    def on_epoch_end(self, state):
        # type: (catalyst.core.State) -> None
        current_score = state.valid_metrics[self.metric]
        self._trial.report(current_score, state.epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(state.epoch)
            raise optuna.exceptions.TrialPruned(message)


def _check_catalyst_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            "Catalyst is not available. Please install Catalyst to use this "
            "feature. Catalyst can be installed by executing `$ pip install "
            "catalyst`. For further information, please refer to the installation guide "
            "of Catalyst. (The actual import error is as follows: "
            + str(_import_error)
            + ")"
        )
