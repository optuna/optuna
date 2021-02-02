from typing import Any

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
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than ``min_delta``, will count as no
            improvement. Default: ``0.0``.
        patience: number of validation epochs with no improvement
            after which training will be stopped. Default: ``3``.
        verbose: verbosity mode. Default: ``False``.
        mode: one of {``min``, ``max``}. In ``min`` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in ``max``
            mode it will stop when the quantity
            monitored has stopped increasing.
        strict: whether to crash the training if ``monitor`` is
            not found in the validation metrics. Default: ``True``.
    """

    def __init__(
        self,
        trial: optuna.trial.Trial,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
    ) -> None:

        _imports.check()

        super(PyTorchLightningPruningCallback, self).__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            strict=strict,
        )

        self._trial = trial

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        # To check if we saved the internal states correctly, moved states from TPU to CPU etc.
        super().on_validation_end(trainer=trainer, pl_module=pl_module)
        logs = trainer.callback_metrics
        epoch = pl_module.current_epoch
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
