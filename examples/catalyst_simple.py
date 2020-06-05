"""
Optuna example that optimizes multi-layer perceptrons using Catalyst.

In this example, we optimize the validation accuracy of hand-written digit recognition using
Catalyst, and MNIST. We optimize the neural network architecture.

We have the following two ways to execute this example:

(1) Execute this code directly. Pruning can be turned on and off with the `--pruning` argument.
    $ python catalyst_simple.py [--pruning]


(2) Execute through CLI. Pruning is enabled automatically.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize catalyst_simple.py objective --n-trials=100 --study \
      $STUDY_NAME --storage sqlite:///example.db
"""

import argparse
import os

from catalyst.dl import AccuracyCallback
from catalyst.dl import SupervisedRunner
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import optuna
from optuna.integration import CatalystPruningCallback


CLASSES = 10


class Net(nn.Module):
    def __init__(self, trial):
        super(Net, self).__init__()
        self.layers = []
        self.dropouts = []

        # We optimize the number of layers, hidden units in each layer and dropouts.
        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_uniform("dropout", 0.2, 0.5)
        input_dim = 28 * 28
        for i in range(n_layers):
            output_dim = int(trial.suggest_loguniform("n_units_l{}".format(i), 4, 128))
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.dropouts.append(nn.Dropout(dropout))
            input_dim = output_dim

        self.layers.append(nn.Linear(input_dim, CLASSES))

        # Assigning the layers as class variables (PyTorch requirement).
        for idx, layer in enumerate(self.layers):
            setattr(self, "fc{}".format(idx), layer)

        # Assigning the dropouts as class variables (PyTorch requirement).
        for idx, dropout in enumerate(self.dropouts):
            setattr(self, "drop{}".format(idx), dropout)

    def forward(self, data):
        data = data.view(-1, 28 * 28)
        for layer, dropout in zip(self.layers, self.dropouts):
            data = F.relu(layer(data))
            data = dropout(data)
        return F.log_softmax(self.layers[-1](data), dim=1)


loaders = {
    "train": DataLoader(
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
        batch_size=100,
    ),
    "valid": DataLoader(
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()),
        batch_size=100,
    ),
}


def objective(trial):
    logdir = "./logdir"
    num_epochs = 10

    model = Net(trial)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = torch.nn.CrossEntropyLoss()

    # model training
    runner = SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loaders=loaders,
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True,
        callbacks=[
            AccuracyCallback(),
            CatalystPruningCallback(
                trial, metric="accuracy01"
            ),  # top-1 accuracy as metric for pruning
        ],
    )

    return runner.state.valid_metrics["accuracy01"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Catalyst example.")
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
    study.optimize(objective, n_trials=10, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
