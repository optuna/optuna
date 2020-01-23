"""
Optuna example that optimizes multi-layer perceptrons using PyTorch Lightning.

In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch Lightning, and MNIST. We optimize the neural network architecture. As it is too time
consuming to use the whole MNIST dataset, we here use a small subset of it.

We have the following two ways to execute this example:

(1) Execute this code directly. Pruning can be turned on and off with the `--pruning` argument.
    $ python pytorch_lightning_simple.py [--pruning]


(2) Execute through CLI. Pruning is enabled automatically.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize pytorch_lightning_simple.py objective --n-trials=100 --study \
      $STUDY_NAME --storage sqlite:///example.db
"""

import argparse
import os
import shutil

import pytorch_lightning as pl
from pytorch_lightning.logging import LightningLoggerBase
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

import optuna
from optuna.integration import PyTorchLightningPruningCallback

PERCENT_TEST_EXAMPLES = 0.1
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 10
DIR = os.getcwd()
MODEL_DIR = os.path.join(DIR, 'result')


class DictLogger(LightningLoggerBase):
    """PyTorch Lightning `dict` logger."""

    def __init__(self, version):
        super(DictLogger, self).__init__()
        self.metrics = []
        self._version = version

    def log_metrics(self, metric, step=None):
        self.metrics.append(metric)

    @property
    def version(self):
        return self._version


class Net(nn.Module):
    def __init__(self, trial):
        super(Net, self).__init__()
        self.layers = []
        self.dropouts = []

        # We optimize the number of layers, hidden untis in each layer and drouputs.
        n_layers = trial.suggest_int('n_layers', 1, 3)
        dropout = trial.suggest_uniform('dropout', 0.2, 0.5)
        input_dim = 28 * 28
        for i in range(n_layers):
            output_dim = int(trial.suggest_loguniform('n_units_l{}'.format(i), 4, 128))
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.dropouts.append(nn.Dropout(dropout))
            input_dim = output_dim

        self.layers.append(nn.Linear(input_dim, CLASSES))

        # Assigning the layers as class variables (PyTorch requirement).
        for idx, layer in enumerate(self.layers):
            setattr(self, 'fc{}'.format(idx), layer)

        # Assigning the dropouts as class variables (PyTorch requirement).
        for idx, dropout in enumerate(self.dropouts):
            setattr(self, 'drop{}'.format(idx), dropout)

    def forward(self, data):
        data = data.view(-1, 28 * 28)
        for layer, dropout in zip(self.layers, self.dropouts):
            data = F.relu(layer(data))
            data = dropout(data)
        return F.log_softmax(self.layers[-1](data), dim=1)


class LightningNet(pl.LightningModule):

    def __init__(self, trial):
        super(LightningNet, self).__init__()

        # Be careful not to overwrite `pl.LightningModule` attributes such as `self.model`.
        self._model = Net(trial)

    def forward(self, data):
        return self._model(data)

    def training_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)
        loss = F.nll_loss(output, target)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / data.size(0)
        return {'validation_accuracy': accuracy}

    def validation_end(self, outputs):
        accuracy = sum(x['validation_accuracy'] for x in outputs) / len(outputs)
        # Pass the accuracy to the `DictLogger` via the `'log'` key.
        return {'log': {'accuracy': accuracy}}

    def configure_optimizers(self):
        return Adam(self._model.parameters())

    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            datasets.MNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=BATCHSIZE, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            datasets.MNIST(DIR, train=False, download=True, transform=transforms.ToTensor()),
            batch_size=BATCHSIZE, shuffle=False)


def objective(trial):
    # PyTorch Lightning will try to restore model parameters from previous trials if checkpoint
    # filenames match. Therefore, the filenames for each trial must be made unique.
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, 'trial_{}'.format(trial.number)), monitor='accuracy')

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.
    logger = DictLogger(trial.number)

    trainer = pl.Trainer(
        logger=logger,
        val_percent_check=PERCENT_TEST_EXAMPLES,
        checkpoint_callback=checkpoint_callback,
        max_epochs=EPOCHS,
        gpus=0 if torch.cuda.is_available() else None,
        early_stop_callback=PyTorchLightningPruningCallback(trial, monitor='accuracy')
    )

    model = LightningNet(trial)
    trainer.fit(model)

    return logger.metrics[-1]['accuracy']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Lightning example.')
    parser.add_argument('--pruning', '-p', action='store_true',
                        help='Activate the pruning feature. `MedianPruner` stops unpromising '
                             'trials at the early stages of training.')
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    shutil.rmtree(MODEL_DIR)
