"""
Optuna example that optimizes multi-layer perceptrons using PyTorch distributed.

In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch distributed data parallel and MNIST. We optimize the neural network architecture as well
as the optimizer configuration. As it is too time consuming to use the whole MNIST dataset, we
here use a small subset of it.

You can execute this example with mpirun command as follows:
    $ mpirun -n 2 -- python pytorch_distributed_simple.py

"""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

import optuna


DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10


def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_float("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_mnist():
    # Load MNIST dataset.
    train_dataset = datasets.MNIST(DIR, train=True, transform=transforms.ToTensor())
    train_dataset = torch.utils.data.Subset(train_dataset, indices=range(N_TRAIN_EXAMPLES))
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=train_dataset)

    valid_dataset = datasets.MNIST(DIR, train=False, transform=transforms.ToTensor())
    valid_dataset = torch.utils.data.Subset(valid_dataset, indices=range(N_VALID_EXAMPLES))
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=valid_dataset, shuffle=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=BATCHSIZE,
        shuffle=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        batch_size=BATCHSIZE,
        shuffle=False,
    )

    return train_loader, valid_loader, train_sampler, valid_sampler


def objective(single_trial):
    trial = optuna.integration.TorchDistributedTrial(single_trial)

    # Generate the model.
    model = DDP(define_model(trial).to(DEVICE))

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Get the MNIST dataset.
    train_loader, valid_loader, train_sampler, valid_sampler = get_mnist()

    accuracy = 0
    # Training of the model.
    for epoch in range(EPOCHS):
        model.train()
        # Shuffle train dataset.
        train_sampler.set_epoch(epoch)
        for data, target in train_loader:
            data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        correct_tensor = torch.tensor([correct], dtype=torch.int)
        dist.all_reduce(correct_tensor)
        total_correct = correct_tensor.numpy().tolist()[0]
        accuracy = total_correct / len(valid_loader.dataset)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":
    # Set environmental variables required by torch.distributed.
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "20000"

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    if rank == 0:
        # Download dataset before starting the optimization.
        datasets.MNIST(DIR, train=True, download=True)
    dist.barrier()

    study = None
    n_trials = 20
    if rank == 0:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
    else:
        for _ in range(n_trials):
            try:
                objective(None)
            except optuna.TrialPruned:
                pass

    if rank == 0:
        assert study is not None
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
