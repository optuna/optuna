"""
Optuna example that optimizes multi-layer perceptrons using Catalyst.

In this example, we optimize the validation accuracy of hand-written digit recognition using
Catalyst, and MNIST. We optimize the neural network architecture.

You can run this example as follows, pruning can be turned on and off with the `--pruning`
argument.
    $ python catalyst_simple.py [--pruning]


See also: https://catalyst-team.github.io/catalyst/api/callbacks.html?highlight=optuna#catalyst.callbacks.optuna.OptunaPruningCallback  # NOQA
"""

import argparse
import os
import urllib

from catalyst.dl import AccuracyCallback
from catalyst.dl import SupervisedRunner
from catalyst.dl import OptunaPruningCallback
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import optuna


# Register a global custom opener to avoid HTTP Error 403: Forbidden when downloading MNIST.
# This is a temporary fix until torchvision v0.9 is released.
opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


CLASSES = 10


def define_model(trial: optuna.trial.Trial) -> nn.Sequential:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    input_dim = 28 * 28
    layers = [nn.Flatten()]
    for i in range(n_layers):
        output_dim = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        input_dim = output_dim
    layers.append(nn.Linear(input_dim, CLASSES))

    return nn.Sequential(*layers)


loaders = {
    "train": DataLoader(
        datasets.MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()),
        batch_size=100,
        shuffle=True,
    ),
    "valid": DataLoader(
        datasets.MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor()),
        batch_size=100,
    ),
}


def objective(trial):
    logdir = "./logdir"
    num_epochs = 2

    model = define_model(trial)
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
        callbacks={
            # top-1 accuracy as metric for pruning
            "optuna": OptunaPruningCallback(
                loader_key="valid",
                metric_key="accuracy01",
                minimize=False,
                trial=trial,
            ),
            "accuracy": AccuracyCallback(
                input_key="logits",
                target_key="targets",
                num_classes=10,
            ),
        },
    )

    return runner.callbacks["optuna"].best_score


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
