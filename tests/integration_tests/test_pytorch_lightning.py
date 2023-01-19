from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import pytest

import optuna
from optuna._imports import try_import
from optuna.integration import PyTorchLightningPruningCallback
from optuna.testing.pruners import DeterministicPruner
from optuna.testing.storages import StorageSupplier


with try_import() as _imports:
    import pytorch_lightning as pl
    from pytorch_lightning import LightningModule
    import torch
    from torch import nn
    import torch.nn.functional as F

if not _imports.is_successful():
    LightningModule = object  # type: ignore[assignment, misc]  # NOQA

pytestmark = pytest.mark.integration


class Model(LightningModule):
    def __init__(self) -> None:

        super().__init__()
        self._model = nn.Sequential(nn.Linear(4, 8))

    def forward(self, data: "torch.Tensor") -> "torch.Tensor":
        return self._model(data)

    def training_step(
        self, batch: List["torch.Tensor"], batch_nb: int
    ) -> Dict[str, "torch.Tensor"]:

        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target)
        return {"loss": loss}

    def validation_step(
        self, batch: List["torch.Tensor"], batch_nb: int
    ) -> Dict[str, "torch.Tensor"]:

        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).double().mean()
        return {"validation_accuracy": accuracy}

    def validation_epoch_end(
        self,
        outputs: Union[
            List[Union["torch.Tensor", Dict[str, Any]]],
            List[List[Union["torch.Tensor", Dict[str, Any]]]],
        ],
    ) -> None:
        if not len(outputs):
            return

        accuracy = sum(
            x["validation_accuracy"] for x in cast(List[Dict[str, "torch.Tensor"]], outputs)
        ) / len(outputs)
        self.log("accuracy", accuracy)

    def configure_optimizers(self) -> "torch.optim.Optimizer":

        return torch.optim.SGD(self._model.parameters(), lr=1e-2)

    def train_dataloader(self) -> "torch.utils.data.DataLoader":

        return self._generate_dummy_dataset()

    def val_dataloader(self) -> "torch.utils.data.DataLoader":

        return self._generate_dummy_dataset()

    def _generate_dummy_dataset(self) -> "torch.utils.data.DataLoader":

        data = torch.zeros(3, 4, dtype=torch.float32)
        target = torch.zeros(3, dtype=torch.int64)
        dataset = torch.utils.data.TensorDataset(data, target)
        return torch.utils.data.DataLoader(dataset, batch_size=1)


class ModelDDP(Model):
    def __init__(self) -> None:

        super().__init__()

    def validation_step(
        self, batch: List["torch.Tensor"], batch_nb: int
    ) -> Dict[str, "torch.Tensor"]:

        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).double().mean()

        if self.local_rank == 0:
            accuracy = torch.tensor(0.3)
        elif self.local_rank == 1:
            accuracy = torch.tensor(0.6)

        self.log("accuracy", accuracy, sync_dist=True)
        return {"validation_accuracy": accuracy}


def test_pytorch_lightning_pruning_callback() -> None:
    def objective(trial: optuna.trial.Trial) -> float:

        trainer = pl.Trainer(
            max_epochs=2,
            enable_checkpointing=False,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="accuracy")],
        )

        model = Model()
        trainer.fit(model)

        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0


def test_pytorch_lightning_pruning_callback_monitor_is_invalid() -> None:

    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = study.ask()
    callback = PyTorchLightningPruningCallback(trial, "InvalidMonitor")

    trainer = pl.Trainer(
        max_epochs=1,
        enable_checkpointing=False,
        callbacks=[callback],
    )
    model = Model()

    with pytest.warns(UserWarning):
        callback.on_validation_end(trainer, model)


@pytest.mark.skip(reason="Currently DDP is not supported")
@pytest.mark.parametrize("storage_mode", ["sqlite", "cached_sqlite"])
def test_pytorch_lightning_pruning_callback_ddp_monitor(
    storage_mode: str,
) -> None:
    def objective(trial: optuna.trial.Trial) -> float:

        trainer = pl.Trainer(
            strategy="ddp",
            accelerator="cpu",
            devices=2,
            num_processes=2,
            enable_checkpointing=False,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="accuracy")],
        )

        model = ModelDDP()
        trainer.fit(model)

        return 1.0

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage, pruner=DeterministicPruner(True))
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state == optuna.trial.TrialState.PRUNED
        assert list(study.trials[0].intermediate_values.keys()) == [0]
        np.testing.assert_almost_equal(study.trials[0].intermediate_values[0], 0.45)

        study = optuna.create_study(storage=storage, pruner=DeterministicPruner(False))
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
        assert study.trials[0].value == 1.0
        assert list(study.trials[0].intermediate_values.keys()) == [0, 1]
        np.testing.assert_almost_equal(study.trials[0].intermediate_values[0], 0.45)
        np.testing.assert_almost_equal(study.trials[0].intermediate_values[1], 0.45)


@pytest.mark.skip(reason="Currently DDP is not supported")
def test_pytorch_lightning_pruning_callback_ddp_unsupported_storage() -> None:
    storage_mode = "inmemory"

    def objective(trial: optuna.trial.Trial) -> float:

        trainer = pl.Trainer(
            max_epochs=1,
            strategy="ddp",
            accelerator="cpu",
            devices=2,
            enable_checkpointing=False,
            callbacks=[PyTorchLightningPruningCallback(trial, monitor="accuracy")],
        )

        model = ModelDDP()
        trainer.fit(model)

        return 1.0

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage, pruner=DeterministicPruner(True))
        with pytest.raises(ValueError):
            study.optimize(objective, n_trials=1)
