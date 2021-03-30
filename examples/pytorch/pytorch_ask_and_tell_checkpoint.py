"""
Optuna example that optimizes multi-layer perceptrons using PyTorch with checkpoint.

In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

Even if the process where the trial is running is killed for some reason, you can restart from
previous saved checkpoint using heartbeat.

    $ timeout 20 python examples/pytorch/pytorch_checkpoint.py
    $ python examples/pytorch/pytorch_checkpoint.py
"""

import copy
import datetime
import os
from threading import TIMEOUT_MAX

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    # Load FashionMNIST dataset.
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(DIR, train=False, transform=transforms.ToTensor()),
        batch_size=BATCHSIZE,
        shuffle=True,
    )

    return train_loader, valid_loader


def objective(trial):

    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    if "checkpoint_path" in trial.user_attrs:
        checkpoint = torch.load(trial.user_attrs["checkpoint_path"])
        epoch_begin = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        accuracy = checkpoint["accuracy"]
    else:
        epoch_begin = 0

    # Get the FashionMNIST dataset.
    train_loader, valid_loader = get_mnist()

    path = f"pytorch_checkpoint/{trial.number}"
    os.makedirs(path, exist_ok=True)

    # Training of the model.
    for epoch in range(epoch_begin, EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

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
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                output = model(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        # Save optimization status. We should save the objective value because the process may be
        # killed between saving the last model and recording the objective value to the storage.
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": accuracy,
            },
            os.path.join(path, "model.pt"),
        )

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def restart_from_checkpoint(study, trial):
    # Enqueue trial with the same parameters as the stale trial to use saved information.

    path = f"pytorch_checkpoint/{trial.number}/model.pt"
    user_attrs = copy.deepcopy(trial.user_attrs)
    if os.path.exists(path):
        user_attrs["checkpoint_path"] = path

    study.add_trial(
        optuna.create_trial(
            state=optuna.trial.TrialState.WAITING,
            params=trial.params,
            distributions=trial.distributions,
            user_attrs=user_attrs,
            system_attrs=trial.system_attrs,
        )
    )


if __name__ == "__main__":
    storage = optuna.storages.RDBStorage(
        "sqlite:///example.db", heartbeat_interval=1, failed_trial_callback=restart_from_checkpoint
    )
    study = optuna.create_study(
        storage=storage, study_name="pytorch_checkpoint", direction="maximize", load_if_exists=True
    )

    N_TRIALS = 10
    TIMEOUT = 600
    start_time = datetime.datetime.now()

    for _ in range(N_TRIALS):
        optuna.study.run_failed_trial_callback(study)
        trial = study.ask()
        thread, stop_event = optuna.study.create_heartbeat_thread(study, trial)

        try:
            value = objective(trial)
            study.tell(trial, value)
            print(
                f"Trial {trial.number} finished with value: {value} and parameters: "
                f"{trial.params}. Best is trial {study.best_trial.number} with "
                f"value: {study.best_value}."
            )
        except optuna.TrialPruned:
            print(f"Trial {trial.number} was pruned.")
            study.tell(trial, None, optuna.trial.TrialState.PRUNED)

        optuna.study.join_heartbeat_thread(study, thread, stop_event)

        duration = (datetime.datetime.now() - start_time).total_seconds()
        if duration >= TIMEOUT:
            break

    pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
    complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

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

    # The line of the resumed trial's intermediate values begins with the restarted epoch.
    optuna.visualization.plot_intermediate_values(study).show()
