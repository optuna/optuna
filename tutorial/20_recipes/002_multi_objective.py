"""
.. _multi_objective:

Multi-objective Optimization with Optuna
========================================

This tutorial showcases Optuna's multi-objective optimization feature by
optimizing the validation accuracy of Fashion MNIST dataset and the FLOPS of the model implemented in PyTorch.

We use `fvcore <https://github.com/facebookresearch/fvcore>`__ to measure FLOPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fvcore.nn import FlopCountAnalysis

import optuna


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DIR = ".."
BATCHSIZE = 128
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float(f"dropout_{i}", 0.2, 0.5)
        layers.append(nn.Dropout(p))

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

    flops = FlopCountAnalysis(model, inputs=(torch.randn(1, 28 * 28).to(DEVICE),)).total()
    return flops, accuracy


###################################################################################################
# Define multi-objective objective function.
# Objectives are FLOPS and accuracy.
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
    flops, accuracy = eval_model(model, val_loader)
    return flops, accuracy


###################################################################################################
# Run multi-objective optimization
# --------------------------------
#
# If your optimization problem is multi-objective,
# Optuna assumes that you will specify the optimization direction for each objective.
# Specifically, in this example, we want to minimize the FLOPS (we want a faster model)
# and maximize the accuracy. So we set ``directions`` to ``["minimize", "maximize"]``.
study = optuna.create_study(directions=["minimize", "maximize"])
study.optimize(objective, n_trials=30, timeout=300)

print("Number of finished trials: ", len(study.trials))


###################################################################################################
# Note that the following sections requires the installation of `Plotly <https://plotly.com/python>`__ for visualization
# and `scikit-learn <https://scikit-learn.org/stable>`__ for hyperparameter importance calculation:
#
# .. code-block:: console
#
#     $ pip install plotly
#     $ pip install scikit-learn
#     $ pip install nbformat  # Required if you are running this tutorial in Jupyter Notebook.
#
# Check trials on Pareto front visually.
optuna.visualization.plot_pareto_front(study, target_names=["FLOPS", "accuracy"])


###################################################################################################
# Fetch the list of trials on the Pareto front with :attr:`~optuna.study.Study.best_trials`.
#
# For example, the following code shows the number of trials on the Pareto front and picks the trial with the highest accuracy.

print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

trial_with_highest_accuracy = max(study.best_trials, key=lambda t: t.values[1])
print("Trial with highest accuracy: ")
print(f"\tnumber: {trial_with_highest_accuracy.number}")
print(f"\tparams: {trial_with_highest_accuracy.params}")
print(f"\tvalues: {trial_with_highest_accuracy.values}")

###################################################################################################
# Learn which hyperparameters are affecting the flops most with hyperparameter importance.
optuna.visualization.plot_param_importances(
    study, target=lambda t: t.values[0], target_name="flops"
)
