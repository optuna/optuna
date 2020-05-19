import optuna

if optuna.type_checking.TYPE_CHECKING:
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA

with optuna._imports.try_import(catch=(ImportError, SyntaxError)) as _imports:
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning import LightningModule
    from pytorch_lightning import Trainer

if not _imports.is_successful():
    EarlyStopping = object  # NOQA


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

        _imports.check()

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
            raise optuna.exceptions.TrialPruned(message)
