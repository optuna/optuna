import optuna


with optuna._imports.try_import() as _imports:
    from pytorch_lightning import LightningModule
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping

if not _imports.is_successful():
    EarlyStopping = object  # NOQA
    LightningModule = object  # NOQA
    Trainer = object  # NOQA


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
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:

        _imports.check()

        super(PyTorchLightningPruningCallback, self).__init__(monitor=monitor)

        self._trial = trial

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        logs = trainer.callback_metrics
        epoch = pl_module.current_epoch
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
