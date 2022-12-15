import warnings

from packaging import version

import optuna
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._rdb.storage import RDBStorage


# Define key names of `Trial.system_attrs`.
_PRUNED_KEY = "ddp_pl:pruned"
_EPOCH_KEY = "ddp_pl:epoch"


with optuna._imports.try_import() as _imports:
    import pytorch_lightning as pl
    from pytorch_lightning import LightningModule
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import Callback

if not _imports.is_successful():
    Callback = object  # type: ignore  # NOQA
    LightningModule = object  # type: ignore  # NOQA
    Trainer = object  # type: ignore  # NOQA


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

    .. note::
        For the distributed data parallel training, the version of PyTorchLightning needs to be
        higher than or equal to v1.5.0. In addition, :class:`~optuna.study.Study` should be
        instantiated with RDB storage.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        _imports.check()
        super().__init__()

        self._trial = trial
        self.monitor = monitor
        self.is_ddp_backend = False

    def on_init_start(self, trainer: Trainer) -> None:
        self.is_ddp_backend = (
            trainer._accelerator_connector.distributed_backend is not None  # type: ignore
        )
        if self.is_ddp_backend:
            if version.parse(pl.__version__) < version.parse("1.5.0"):  # type: ignore
                raise ValueError("PyTorch Lightning>=1.5.0 is required in DDP.")
            if not (
                isinstance(self._trial.study._storage, _CachedStorage)
                and isinstance(self._trial.study._storage._backend, RDBStorage)
            ):
                raise ValueError(
                    "optuna.integration.PyTorchLightningPruningCallback"
                    " supports only optuna.storages.RDBStorage in DDP."
                )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        should_stop = False
        if trainer.is_global_zero:
            self._trial.report(current_score.item(), step=epoch)
            should_stop = self._trial.should_prune()
        should_stop = trainer.training_type_plugin.broadcast(should_stop)
        if not should_stop:
            return

        if not self.is_ddp_backend:
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
        else:
            # Stop every DDP process if global rank 0 process decides to stop.
            trainer.should_stop = True
            if trainer.is_global_zero:
                self._trial.storage.set_trial_system_attr(self._trial._trial_id, _PRUNED_KEY, True)
                self._trial.storage.set_trial_system_attr(self._trial._trial_id, _EPOCH_KEY, epoch)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.is_ddp_backend:
            return

        # Because on_validation_end is executed in spawned processes,
        # _trial.report is necessary to update the memory in main process, not to update the RDB.
        _trial_id = self._trial._trial_id
        _study = self._trial.study
        _trial = _study._storage._backend.get_trial(_trial_id)  # type: ignore
        is_pruned = _trial.system_attrs.get(_PRUNED_KEY)
        epoch = _trial.system_attrs.get(_EPOCH_KEY)
        intermediate_values = _trial.intermediate_values
        for step, value in intermediate_values.items():
            self._trial.report(value, step=step)

        if is_pruned:
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
