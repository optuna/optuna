"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python pytorch_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --storage sqlite:///example.db`
    $ optuna study optimize pytorch_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db

"""

from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

DEVICE = torch.device('cpu')
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_TEST_EXAMPLES = BATCHSIZE * 10


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


def get_mnist():
    # Load MNIST dataset.
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(DIR,
                                                              train=True,
                                                              download=True,
                                                              transform=transforms.ToTensor()),
                                               batch_size=BATCHSIZE,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(DIR,
                                                             train=False,
                                                             transform=transforms.ToTensor()),
                                              batch_size=BATCHSIZE,
                                              shuffle=True)

    return train_loader, test_loader


def objective(trial):

    # Generate the model.
    model = Net(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_uniform('lr', 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the MNIST dataset.
    train_loader, test_loader = get_mnist()

    # Training of the model.
    model.train()
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.to(DEVICE), target.to(DEVICE)

            # Zeroing out gradient buffers.
            optimizer.zero_grad()
            # Performing a forward pass.
            output = model(data)
            # Computing negative Log Likelihood loss.
            loss = F.nll_loss(output, target)
            # Performing a backward pass.
            loss.backward()
            # Updating the weights.
            optimizer.step()

    # Validation of the model.
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Limiting testing data.
            if batch_idx * BATCHSIZE >= N_TEST_EXAMPLES:
                break
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability.
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct / N_TEST_EXAMPLES
    return accuracy


if __name__ == '__main__':
    import optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
