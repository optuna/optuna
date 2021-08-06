import warnings

import sqlalchemy

import optuna

with optuna._imports.try_import() as _imports:
    from pytorch_lightning import LightningModule
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import Callback

if not _imports.is_successful():
    Callback = object  # type: ignore # NOQA
    LightningModule = object  # type: ignore # NOQA
    Trainer = object  # type: ignore # NOQA


class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
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
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)


class PyTorchLightningDDPPruningCallback(Callback):
    """PyTorch Lightning DDP callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
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
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        # When multi gpu, on_validation_end function executes multi times.
        # So Exception is ignored to avoid IntegrityError.
        distributed_backend = trainer.accelerator_connector.distributed_backend
        try:
            self._trial.report(current_score, step=epoch)
        except sqlalchemy.exc.IntegrityError:
            if distributed_backend is not None:
                pass
            else:
                raise

        if self._trial.should_prune():
            trainer.should_stop = True
            try:
                self._trial.set_user_attr("pruned", True)
                self._trial.set_user_attr("epoch", epoch)
            except sqlalchemy.exc.IntegrityError:
                if distributed_backend is not None:
                    pass
                else:
                    raise

    # Because on_validation_end is executed in threads,  when on_fit_end is executed, it is necessary to report an objective function value for a given step via RDB.
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        _trial_id = self._trial._trial_id
        _study = self._trial.study
        _trial = _study._storage._backend.get_trial(_trial_id)
        is_pruned = _trial.user_attrs.get("pruned")
        epoch = _trial.user_attrs.get("epoch")
        intermediate_values = _trial.intermediate_values
        for step , value in intermediate_values.items():
            self._trial.report(value, step=step)
        
        if is_pruned:
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
