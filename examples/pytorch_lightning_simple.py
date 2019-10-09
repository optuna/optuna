"""
Optuna example that optimizes multi-layer perceptrons using PyTorch Lightning.

In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch Lightning, and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python pytorch_lightning_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize pytorch_lightning_simple.py objective --n-trials=100 --study \
      $STUDY_NAME --storage sqlite:///example.db

"""

from __future__ import division
from __future__ import print_function

import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

import optuna

PERCENT_TRAIN_EXAMPLES = 0.1
PERCENT_TEST_EXAMPLES = 0.1
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 10
DIR = os.getcwd()


class DictLogger(pl.logging.LightningLoggerBase):
    """PyTorch Lightning `dict` logger."""

    def __init__(self):
        super(DictLogger, self).__init__()
        self.metrics = []

    def log_metrics(self, metric, step_num=None):
        self.metrics.append(metric)


# The code for `Net` is taken from `pytorch_simple.py`.
class Net(nn.Module):
    # Constructor for trial network.
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

    # Forward pass computation function.
    def forward(self, data):
        data = data.view(-1, 28 * 28)
        for layer, dropout in zip(self.layers, self.dropouts):
            data = F.relu(layer(data))
            data = dropout(data)
        return F.log_softmax(self.layers[-1](data), dim=1)


class LightningNet(pl.LightningModule):

    def __init__(self, model, optimizer_name, lr):
        super(LightningNet, self).__init__()

        # Be careful not to overwrite `pl.LightningModule` attributes such as `self.model`.
        self._model = model
        self._optimizer_name = optimizer_name
        self._lr = lr

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
        return getattr(optim, self._optimizer_name)(self._model.parameters(), lr=self._lr)

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
        os.path.join(DIR, 'results', 'trial_{}'.format(trial.number)))

    # The default logger in PyTorch Lightining writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.
    logger = DictLogger()

    trainer = pl.Trainer(
        logger=logger,
        train_percent_check=PERCENT_TRAIN_EXAMPLES,
        val_percent_check=PERCENT_TEST_EXAMPLES,
        checkpoint_callback=checkpoint_callback,
        max_nb_epochs=EPOCHS,
        gpus=0 if torch.cuda.is_available() else None,
    )

    # Generate the model.
    model = Net(trial)

    # Sample optimizer parameters.
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_uniform('lr', 1e-5, 1e-1)

    # Generate the PyTorch Lightning model.
    model = LightningNet(model, optimizer_name, lr)
    trainer.fit(model)

    return logger.metrics[-1]['accuracy']


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
