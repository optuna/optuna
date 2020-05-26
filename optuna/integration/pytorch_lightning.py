import optuna

if optuna.type_checking.TYPE_CHECKING:
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA

try:
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning import LightningModule
    from pytorch_lightning import Trainer

    _available = True
except (ImportError, SyntaxError) as e:
    # SyntaxError is raised with Python versions below 3.6 since PyTorch Lightning does not
    # support them.
    _import_error = e
    # PyTorchLightningPruningCallback is disabled because PyTorch Lightning is not available.
    _available = False
    EarlyStopping = object


class PyTorchLightningPruningCallback(EarlyStopping):
    """PyTorch Lightning callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial, monitor):
        # type: (optuna.trial.Trial, str) -> None

        super(PyTorchLightningPruningCallback, self).__init__(monitor=monitor)

        _check_pytorch_lightning_availability()

        self._trial = trial
        self._monitor = monitor

    def on_epoch_end(self, trainer, pl_module):
        # type: (Trainer, LightningModule) -> None
        logs = trainer.callback_metrics
        epoch = pl_module.current_epoch
        current_score = logs.get(self._monitor)
        if current_score is None:
            return
        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)


def _check_pytorch_lightning_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            "PyTorch Lightning is not available. Please install PyTorch Lightning to use this "
            "feature. PyTorch Lightning can be installed by executing `$ pip install "
            "pytorch-lightning`. For further information, please refer to the installation guide "
            "of PyTorch Lightning. (The actual import error is as follows: "
            + str(_import_error)
            + ")"
        )
