"""
Optuna example that optimizes multi-layer perceptrons using PyTorch Lightning.

In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch Lightning, and MNIST. We optimize the neural network architecture. As it is too time
consuming to use the whole MNIST dataset, we here use a small subset of it.

You can run this example as follows, pruning can be turned on and off with the `--pruning`
argument.
    $ python pytorch_lightning_simple.py [--pruning]

"""

import argparse
import os
import shutil

from packaging import version
import pytorch_lightning as pl
from pytorch_lightning import Callback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

import optuna
from optuna.integration import PyTorchLightningPruningCallback


if version.parse(pl.__version__) < version.parse("0.8.1"):
    raise RuntimeError("PyTorch Lightning>=0.8.1 is required for this example.")

PERCENT_VALID_EXAMPLES = 0.1
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 10
DIR = os.getcwd()
MODEL_DIR = os.path.join(DIR, "result")


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class Net(nn.Module):
    def __init__(self, trial):
        super(Net, self).__init__()
        self.layers = []
        self.dropouts = []

        # We optimize the number of layers, hidden units in each layer and dropouts.
        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        input_dim = 28 * 28
        for i in range(n_layers):
            output_dim = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.dropouts.append(nn.Dropout(dropout))
            input_dim = output_dim

        self.layers.append(nn.Linear(input_dim, CLASSES))

        # Assigning the layers as class variables (PyTorch requirement).
        # Parameters of a layer are returned when calling model.parameters(),
        # only if the layer is a class variable. Thus, assigning as class
        # variable is necessary to make the layer parameters trainable.
        for idx, layer in enumerate(self.layers):
            setattr(self, "fc{}".format(idx), layer)

        # Assigning the dropouts as class variables (PyTorch requirement), for
        # the same reason as above.
        for idx, dropout in enumerate(self.dropouts):
            setattr(self, "drop{}".format(idx), dropout)

    def forward(self, data):
        data = data.view(-1, 28 * 28)
        for layer, dropout in zip(self.layers, self.dropouts):
            data = F.relu(layer(data))
            data = dropout(data)
        return F.log_softmax(self.layers[-1](data), dim=1)


class LightningNet(pl.LightningModule):
    def __init__(self, trial):
        super(LightningNet, self).__init__()
        self.model = Net(trial)

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)
        return {"loss": F.nll_loss(output, target)}

    def validation_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        return {"batch_val_acc": accuracy}

    def validation_epoch_end(self, outputs):
        accuracy = sum(x["batch_val_acc"] for x in outputs) / len(outputs)
        # Pass the accuracy to the `DictLogger` via the `'log'` key.
        return {"log": {"val_acc": accuracy}}

    def configure_optimizers(self):
        return Adam(self.model.parameters())

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            datasets.MNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=BATCHSIZE,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            datasets.MNIST(DIR, train=False, download=True, transform=transforms.ToTensor()),
            batch_size=BATCHSIZE,
            shuffle=False,
        )


def objective(trial):
    # Filenames for each trial must be made unique in order to access each checkpoint.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, "trial_{}".format(trial.number), "{epoch}"), monitor="val_acc"
    )

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We don't use any logger here as it requires us to implement several abstract
    # methods. Instead we setup a simple callback, that saves metrics from each validation step.
    metrics_callback = MetricsCallback()
    trainer = pl.Trainer(
        logger=False,
        limit_val_batches=PERCENT_VALID_EXAMPLES,
        checkpoint_callback=checkpoint_callback,
        max_epochs=EPOCHS,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[metrics_callback, PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )

    model = LightningNet(trial)
    trainer.fit(model)

    return metrics_callback.metrics[-1]["val_acc"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Lightning example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    shutil.rmtree(MODEL_DIR)
