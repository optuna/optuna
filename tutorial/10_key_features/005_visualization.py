"""
.. _visualization:

Quick Visualization for Hyperparameter Optimization Analysis
============================================================

Optuna provides various visualization features in :mod:`optuna.visualization` to analyze optimization results visually.

This tutorial walks you through this module by visualizing the optimization results of PyTorch model for FashionMNIST dataset.

For visualizing multi-objective optimization (i.e., the usage of :func:`optuna.visualization.plot_pareto_front`),
please refer to the tutorial of :ref:`multi_objective`.

.. note::
   By using `Optuna Dashboard <https://github.com/optuna/optuna-dashboard>`_, you can also check the optimization history,
   hyperparameter importances, hyperparameter relationships, etc. in graphs and tables.
   Please make your study persistent using :ref:`RDB backend <rdb>` and execute following commands to run Optuna Dashboard.

   .. code-block:: console

      $ pip install optuna-dashboard
      $ optuna-dashboard sqlite:///example-study.db

   Please check out `the GitHub repository <https://github.com/optuna/optuna-dashboard>`_ for more details.

   .. list-table::
      :header-rows: 1

      * - Manage Studies
        - Visualize with Interactive Graphs
      * - .. image:: https://user-images.githubusercontent.com/5564044/205545958-305f2354-c7cd-4687-be2f-9e46e7401838.gif
        - .. image:: https://user-images.githubusercontent.com/5564044/205545965-278cd7f4-da7d-4e2e-ac31-6d81b106cada.gif
"""

###################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


import optuna

# You can use Matplotlib instead of Plotly for visualization by simply replacing `optuna.visualization` with
# `optuna.visualization.matplotlib` in the following examples.
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_rank
from optuna.visualization import plot_slice
from optuna.visualization import plot_timeline


SEED = 13
torch.manual_seed(SEED)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DIR = ".."
BATCHSIZE = 128
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 2)
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 64, 512)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())

        in_features = out_features

    layers.append(nn.Linear(in_features, 10))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


# Defines training and evaluation.
def train_model(model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        F.nll_loss(model(data), target).backward()
        optimizer.step()


def eval_model(model, valid_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(valid_loader):
            data, target = data.view(-1, 28 * 28).to(DEVICE), target.to(DEVICE)
            pred = model(data).argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / N_VALID_EXAMPLES

    return accuracy


###################################################################################################
# Define the objective function.
def objective(trial):
    train_dataset = torchvision.datasets.FashionMNIST(
        DIR, train=True, download=True, transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, list(range(N_TRAIN_EXAMPLES))),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    val_dataset = torchvision.datasets.FashionMNIST(
        DIR, train=False, transform=torchvision.transforms.ToTensor()
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(val_dataset, list(range(N_VALID_EXAMPLES))),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    model = define_model(trial).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    )

    for epoch in range(10):
        train_model(model, optimizer, train_loader)

        val_accuracy = eval_model(model, val_loader)
        trial.report(val_accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_accuracy


###################################################################################################
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(),
)
study.optimize(objective, n_trials=30, timeout=300)

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
plot_parallel_coordinate(study, params=["lr", "n_layers"])

###################################################################################################
# Visualize hyperparameter relationships. See :func:`~optuna.visualization.plot_contour` for the details.
plot_contour(study)

###################################################################################################
# Select parameters to visualize.
plot_contour(study, params=["lr", "n_layers"])

###################################################################################################
# Visualize individual hyperparameters as slice plot. See :func:`~optuna.visualization.plot_slice` for the details.
plot_slice(study)

###################################################################################################
# Select parameters to visualize.
plot_slice(study, params=["lr", "n_layers"])

###################################################################################################
# Visualize parameter importances. See :func:`~optuna.visualization.plot_param_importances` for the details.
plot_param_importances(study)

###################################################################################################
# Learn which hyperparameters are affecting the trial duration with hyperparameter importance.
optuna.visualization.plot_param_importances(
    study, target=lambda t: t.duration.total_seconds(), target_name="duration"
)

###################################################################################################
# Visualize empirical distribution function. See :func:`~optuna.visualization.plot_edf` for the details.
plot_edf(study)

###################################################################################################
# Visualize parameter relations with scatter plots colored by objective values. See :func:`~optuna.visualization.plot_rank` for the details.
plot_rank(study)

###################################################################################################
# Visualize the optimization timeline of performed trials. See :func:`~optuna.visualization.plot_timeline` for the details.
plot_timeline(study)

###################################################################################################
# Customize generated figures
# ---------------------------
# In :mod:`optuna.visualization` and :mod:`optuna.visualization.matplotlib`, a function returns an editable figure object:
# :class:`plotly.graph_objects.Figure` or :class:`matplotlib.axes.Axes` depending on the module.
# This allows users to modify the generated figure for their demand by using API of the visualization library.
# The following example replaces figure titles drawn by Plotly-based :func:`~optuna.visualization.plot_intermediate_values` manually.
fig = plot_intermediate_values(study)

fig.update_layout(
    title="Hyperparameter optimization for FashionMNIST classification",
    xaxis_title="Epoch",
    yaxis_title="Validation Accuracy",
)
