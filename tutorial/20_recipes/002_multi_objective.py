"""
.. _multi_objective:

Multi-objective Optimization with Optuna
========================================

This tutorial showcases Optuna's multi-objective optimization feature by
optimizing the validation accuracy of Fashion MNIST dataset and the FLOPS of the model implemented in PyTorch.

We use `thop <https://github.com/Lyken17/pytorch-OpCounter>`_ to measure FLOPS.
"""

import thop
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_{}".format(i), 0.2, 0.5)
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

    flops, _ = thop.profile(model, inputs=(torch.randn(1, 28 * 28).to(DEVICE),), verbose=False)
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
# Check trials on pareto front visually
optuna.visualization.plot_pareto_front(study, target_names=["FLOPS", "accuracy"])


###################################################################################################
# Learn which hyperparameters are affecting the flops most with hyperparameter importance.
optuna.visualization.plot_param_importances(
    study, target=lambda t: t.values[0], target_name="flops"
)
