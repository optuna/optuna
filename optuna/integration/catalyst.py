import optuna

try:
    from catalyst.dl import Callback

    _available = True
except ImportError as e:
    _import_error = e
    # CatalystPruningCallback is disabled because Catalyst is not available.
    _available = False
    Callback = object

from catalyst.dl import Callback

class CatalystPruningCallback(Callback):
    """Catalyst callback to prune unpromising trials.
    Example:
        Add a pruning callback.
        .. code::

            from optuna.integration.catalyst import CatalystPruningCallback
            from catalyst.dl import SupervisedRunner

            runner = SupervisedRunner()
            runner.train(
                model=model,
                criterion=nn.NLLLoss(), # a bit different loss compute
                optimizer=optimizer,
                scheduler=scheduler,
                loaders={'train': train_loader, 'valid': valid_loader},
                logdir="./logs/cv",
                num_epochs=num_epochs,
                verbose=True,
                callbacks=[CatalystPruningCallback()]
            )
    Args:
    """

    def __init__(self, trial, monitor):
        pass

    def on_epoch_end(self, epoch):
        pass

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
