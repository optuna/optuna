"""
.. _visualization:

Quick Visualization for Hyperparameter Optimization Analysis
============================================================

Optuna provides various visualization features in :mod:`optuna.visualization` to optimization results visually.

This tutorial walks you through this module by visualizing the history of multi-layer preceptron for FashionMNIST implemented in PyTorch.
"""

###################################################################################################
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

SEED = 42
BATCH_SIZE = 256
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
DIR = ".."
# Reduce the number of samples for faster build.
N_TRAIN_SAMPLES = BATCH_SIZE * 30
N_VALID_SAMPLES = BATCH_SIZE * 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
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
# Run hyperparameter optimization with :class:`optuna.pruners.MedianPruner`.
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(objective, n_trials=100, timeout=600)

###################################################################################################
# Plot functions
# --------------
# Visualize the optimization history. See :func:`~optuna.visualization.plot_optimization_history` for the details.
plot_optimization_history(study)

###################################################################################################
# Visualize the learning curves of the trials. See :func:`~optuna.visualization.plot_intermediate_values` for the details.
plot_intermediate_values(study)

###################################################################################################
# Visualize high-dimensional parameter relationships. See :func:`~optuna.visualization.plot_parallel_coordinate` for the details.
plot_parallel_coordinate(study)

###################################################################################################
# Select parameters to visualize.
plot_parallel_coordinate(study, params=["lr_init", "n_units_l0"])

###################################################################################################
# Visualize hyperparameter relationships. See :func:`~optuna.visualization.plot_contour` for the details.
plot_contour(study)

###################################################################################################
# Select parameters to visualize.
plot_contour(study, params=["n_units_l0", "n_units_l1"])

###################################################################################################
# Visualize individual hyperparameters as slice plot. See :func:`~optuna.visualization.plot_slice` for the details.
plot_slice(study)

###################################################################################################
# Select parameters to visualize.
plot_slice(study, params=["n_units_l0", "n_units_l1"])

###################################################################################################
# Visualize parameter importances. See :func:`~optuna.visualization.plot_param_importances` for the details.
plot_param_importances(study)

###################################################################################################
# Visualize empirical distribution function. See :func:`~optuna.visualization.plot_edf` for the details.
plot_edf(study)
