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
    $ optuna study optimize mxnet_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db

"""

from __future__ import print_function

import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
BATCHSIZE = 128
CLASSES = 10
DIR = tempfile.mkdtemp()
EPOCHS = 10
LOG_INTERVAL = 10


class Net(nn.Module):
    def __init__(self, trial):
        super(Net, self).__init__()
        self.layers = []
        self.dropouts = []

        n_layers = trial.suggest_int('n_layers', 1, 3)
        dropout = trial.suggest_uniform('dropout', 0.2, 0.5)
        input_dim = 28*28
        for i in range(n_layers):
            output_dim = int(trial.suggest_loguniform('n_units_l{}'.format(i), 4, 128))
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.dropouts.append(nn.Dropout(dropout))
            input_dim = output_dim

        self.layers.append(nn.Linear(input_dim, CLASSES))

        for idx, layer in enumerate(self.layers):
            setattr(self, 'fc{}'.format(idx), layer)

        for idx, dropout in enumerate(self.dropouts):
            setattr(self, 'drop{}'.format(idx), dropout)

    def forward(self, data):
        data = data.view(-1, 28*28)
        for layer, dropout in zip(self.layers, self.dropouts):
            data = F.relu(layer(data))
            # data = dropout(data)
        return F.log_softmax(self.layers[-1](data), dim=1)


def get_mnist():
    # Load MNIST dataset
    kwargs = {'num_workers': 1, 'pin_memory': True} if DEVICE.type == 'cuda' else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCHSIZE, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DIR, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCHSIZE, shuffle=True, **kwargs)

    return (train_loader, test_loader)


def objective(trial):

    # Generate the model
    model = Net(trial).to(DEVICE)

    # Generate the optimizers
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_uniform('lr', 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the MNIST dataset
    train_loader, test_loader = get_mnist()

    # Train the model
    model.train()
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            # Zero gradient buffers
            optimizer.zero_grad()
            # Forward pass
            output = model(data)
            # Negative Log Likelihood loss
            loss = F.nll_loss(output, target)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()

            # Printing statistics
            if batch_idx % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    # Test the model
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

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
