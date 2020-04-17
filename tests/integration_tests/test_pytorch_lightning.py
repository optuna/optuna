import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.testing.integration import DeterministicPruner
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Union  # NOQA


class Model(pl.LightningModule):
    def __init__(self):
        # type: () -> None

        super(Model, self).__init__()
        self._model = nn.Sequential(nn.Linear(4, 8))

    def forward(self, data):
        # type: (torch.Tensor) -> torch.Tensor

        return self._model(data)

    def training_step(self, batch, batch_nb):
        # type: (List[torch.Tensor], int) -> Dict[str, torch.Tensor]

        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target)
        return {"loss": loss}

    def validation_step(self, batch, batch_nb):
        # type: (List[torch.Tensor], int) -> Dict[str, torch.Tensor]

        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum()
        accuracy = correct.double() / data.size(0)
        return {"validation_accuracy": accuracy}

    def validation_end(self, outputs):
        # type: (List[Dict[str, torch.Tensor]]) -> Dict[str, Union[torch.Tensor, float]]

        accuracy = sum(x["validation_accuracy"] for x in outputs) / len(outputs)
        return {"accuracy": accuracy}

    def configure_optimizers(self):
        # type: () -> torch.optim.Optimizer

        return torch.optim.SGD(self._model.parameters(), lr=1e-2)

    @pl.data_loader
    def train_dataloader(self):
        # type: () -> torch.utils.data.DataLoader

        return self._generate_dummy_dataset()

    @pl.data_loader
    def val_dataloader(self):
        # type: () -> torch.utils.data.DataLoader

        return self._generate_dummy_dataset()

    def _generate_dummy_dataset(self):
        # type: () -> torch.utils.data.DataLoader

        data = torch.zeros(3, 4, dtype=torch.float32)
        target = torch.zeros(3, dtype=torch.int64)
        dataset = torch.utils.data.TensorDataset(data, target)
        return torch.utils.data.DataLoader(dataset, batch_size=1)


def test_pytorch_lightning_pruning_callback():
    # type: () -> None

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        trainer = pl.Trainer(
            early_stop_callback=PyTorchLightningPruningCallback(trial, monitor="accuracy"),
            min_nb_epochs=0,  # Required to fire the callback after the first epoch.
            max_nb_epochs=2,
        )
        trainer.checkpoint_callback = None  # Disable unrelated checkpoint callbacks.

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
