"""
Optuna example that optimizes multi-layer perceptrons using PyTorch with Horovod.

In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.

Pytorch with Horovod is supposed to be invoked via MPI. You can run this example as follows:
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ mpirun -n 2 -- pytorch_horovod.py $STUDY_NAME sqlite:///example.db

"""
import sys

import horovod.torch as hvd
from mpi4py import MPI
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets
from torchvision import transforms

import optuna

DEVICE = torch.device("cpu")
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 6


def define_model(trial):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []

    in_features = 28 * 28
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_uniform("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)


def get_mnist():
    # Load MNIST dataset.
    kwargs = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (
        kwargs.get("num_workers", 0) > 0
        and hasattr(mp, "_supports_context")
        and mp._supports_context
        and "forkserver" in mp.get_all_start_methods()
    ):
        kwargs["multiprocessing_context"] = "forkserver"

    train_dataset = datasets.MNIST(
        "data-%d" % hvd.rank(), train=True, download=True, transform=transforms.ToTensor()
    )
    # Horovod: use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCHSIZE, sampler=train_sampler, **kwargs
    )

    test_dataset = datasets.MNIST(
        "data-%d" % hvd.rank(), train=False, transform=transforms.ToTensor()
    )
    # Horovod: use DistributedSampler to partition the test data.
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    valid_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCHSIZE, sampler=valid_sampler, **kwargs
    )
    return train_loader, train_sampler, valid_loader, valid_sampler


def objective(trial, comm):

    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    compression = hvd.Compression.none

    optimizer = hvd.DistributedOptimizer(
        optimizer,
        named_parameters=model.named_parameters(),
        compression=compression,
        op=hvd.Average,
    )

    # Get the MNIST dataset.
    train_loader, _, valid_loader, valid_sampler = get_mnist()

    # Training of the model.
    model.train()
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
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
            for batch_idx, (data, target) in enumerate(valid_loader):
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(valid_sampler)
        accuracy_tensor = torch.tensor(accuracy)
        accuracy_tensor = hvd.allreduce(accuracy_tensor, name="avg_accuracy")
        accuracy = accuracy_tensor.item()

        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


if __name__ == "__main__":

    hvd.init()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())

    torch.set_num_threads(1)

    study_name = sys.argv[1]
    storage_url = sys.argv[2]
    study = optuna.load_study(study_name, storage_url)

    comm = MPI.COMM_WORLD
    mpi_study = optuna.integration.MPIStudy(study, comm)
    mpi_study.optimize(objective, n_trials=20)

    if comm.rank == 0:
        print("Number of finished trials: ", len(mpi_study.trials))

        print("Best trial:")
        trial = mpi_study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
