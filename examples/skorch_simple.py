"""
Optuna example that optimizes multi-layer perceptrons using skorch.

In this example, we optimize the validation accuracy of hand-written digit recognition using
skorch, and MNIST. We optimize the neural network architecture. As it is too time
consuming to use the whole MNIST dataset, we here use a small subset of it.

You can run this example as follows, pruning can be turned on and off with the `--pruning`
argument.
    $ python skorch_simple.py [--pruning]

"""

import argparse

import numpy as np
import optuna
from optuna.integration import SkorchPruningCallback
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import skorch

SUBSET_RATIO = 0.4

mnist = fetch_openml("mnist_784", cache=False)

X = mnist.data.astype('float32')
y = mnist.target.astype('int64')
indices = np.random.permutation(len(X))
N = int(len(X) * SUBSET_RATIO)
X = X[indices][:N]
y = y[indices][:y]

X /= 255.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ClassifierModule(nn.Module):

    def __init__(self, trial: optuna.Trial) -> None:
        super().__init__()

        # We optimize the number of layers, hidden units in each layer and dropouts.
        layers = []
        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        input_dim = 28 * 28
        for i in range(n_layers):
            output_dim = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
            input_dim = output_dim

        layers.append(nn.Linear(input_dim, 10))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.model(x), dim=-1)


def objective(trial: optuna.Trial) -> float:
    net = skorch.NeuralNetClassifier(
        ClassifierModule(trial),
        max_epochs=20,
        lr=0.1,
        device=device,
        callbacks=[SkorchPruningCallback(trial, 'valid_acc')],
    )

    net.fit(X_train, y_train)

    return accuracy_score(y_test, net.predict(X_test))


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

    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
