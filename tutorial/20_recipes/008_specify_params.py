"""
.. _specify_params:

Specify Hyperparameters
=======================

Sometimes, it is natural for you to try some experiments with your out-of-box hyperparameters.
For example, you want to try ResNet50 and Adam with some specific parameters and Gradient Boosting with some minimum samples per node before letting Optuna search exclusively the optimal hyperparameters.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import optuna


BATCH_SIZE = 256
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DIR = ".."
# Reduce the number of samples for faster build.
N_TRAIN_SAMPLES = BATCH_SIZE * 30
N_VALID_SAMPLES = BATCH_SIZE * 10


###################################################################################################
# Before defining the objective function, prepare some utility functions for training.
def train_model(model, optimizer, train_loader):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * BATCH_SIZE >= N_TRAIN_SAMPLES:
            break
        data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def eval_model(model, valid_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            if batch_idx * BATCH_SIZE >= N_VALID_SAMPLES:
                break
            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / min(len(valid_loader.dataset), N_VALID_SAMPLES)
    return accuracy


###################################################################################################
# Define the objective function.
def objective(trial):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(
            DIR, train=True, download=True, transform=torchvision.transforms.ToTensor()
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST(
            DIR, train=False, download=True, transform=torchvision.transforms.ToTensor()
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    layers = []
    in_features = 28 * 28
    for i in range(3):
        # Optimize the number of units of each layer and the initial learning rate.
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 256)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        in_features = out_features
    layers.append(nn.Linear(in_features, 10))
    layers.append(nn.LogSoftmax(dim=1))
    model = nn.Sequential(*layers).to(DEVICE)
    # Sample the initial learning rate from [1e-5, 1e-1] in log space.
    optimizer = torch.optim.Adam(
        model.parameters(), trial.suggest_float("lr_init", 1e-5, 1e-1, log=True)
    )

    for step in range(10):
        model.train()
        train_model(model, optimizer, train_loader)

        accuracy = eval_model(model, valid_loader)

        # Report intermediate objective value.
        trial.report(accuracy, step)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

    return accuracy


###################################################################################################
# You have some sets of hyperparameters that you want to try.
# Then, :func:`~optuna.study.Study.enqueue_trial` does the thing.

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.enqueue_trial(
    {
        "n_units_l0": 50,
        "n_units_l1": 50,
        "n_units_l2": 50,
        "lr_init": 1e-3,
    }
)
study.enqueue_trial(
    {
        "n_units_l0": 100,
        "n_units_l1": 100,
        "n_units_l2": 100,
        "lr_init": 1e-3,
    }
)
study.enqueue_trial(
    {
        "n_units_l0": 200,
        "n_units_l1": 200,
        "n_units_l2": 200,
        "lr_init": 1e-3,
    }
)
import logging
import sys

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study.optimize(objective, n_trials=100, timeout=600)
