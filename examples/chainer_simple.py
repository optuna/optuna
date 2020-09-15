"""
Optuna example that optimizes multi-layer perceptrons using Chainer.

In this example, we optimize the validation accuracy of hand-written digit recognition using
Chainer and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.

"""

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from packaging import version

import optuna
from optuna.integration import ChainerPruningExtension


if version.parse(chainer.__version__) < version.parse("4.0.0"):
    raise RuntimeError("Chainer>=4.0.0 is required for this example.")

N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000
BATCHSIZE = 128
EPOCH = 10


def create_model(trial):
    # We optimize the numbers of layers and their units.
    n_layers = trial.suggest_int("n_layers", 1, 3)

    layers = []
    for i in range(n_layers):
        n_units = int(trial.suggest_float("n_units_l{}".format(i), 4, 128, log=True))
        layers.append(L.Linear(None, n_units))
        layers.append(F.relu)
    layers.append(L.Linear(None, 10))

    return chainer.Sequential(*layers)


def create_optimizer(trial, model):
    # We optimize the choice of optimizers as well as their parameters.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "MomentumSGD"])
    if optimizer_name == "Adam":
        adam_alpha = trial.suggest_float("adam_alpha", 1e-5, 1e-1, log=True)
        optimizer = chainer.optimizers.Adam(alpha=adam_alpha)
    else:
        momentum_sgd_lr = trial.suggest_float("momentum_sgd_lr", 1e-5, 1e-1, log=True)
        optimizer = chainer.optimizers.MomentumSGD(lr=momentum_sgd_lr)

    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    return optimizer


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    # Model and optimizer
    model = L.Classifier(create_model(trial))
    optimizer = create_optimizer(trial, model)

    # Dataset
    rng = np.random.RandomState(0)
    train, valid = chainer.datasets.get_mnist()
    train = chainer.datasets.SubDataset(
        train, 0, N_TRAIN_EXAMPLES, order=rng.permutation(len(train))
    )
    valid = chainer.datasets.SubDataset(
        valid, 0, N_VALID_EXAMPLES, order=rng.permutation(len(valid))
    )
    train_iter = chainer.iterators.SerialIterator(train, BATCHSIZE)
    valid_iter = chainer.iterators.SerialIterator(valid, BATCHSIZE, repeat=False, shuffle=False)

    # Trainer
    updater = chainer.training.StandardUpdater(train_iter, optimizer)
    trainer = chainer.training.Trainer(updater, (EPOCH, "epoch"))
    trainer.extend(chainer.training.extensions.Evaluator(valid_iter, model))
    log_report_extension = chainer.training.extensions.LogReport(log_name=None)
    trainer.extend(
        chainer.training.extensions.PrintReport(
            [
                "epoch",
                "main/loss",
                "validation/main/loss",
                "main/accuracy",
                "validation/main/accuracy",
            ]
        )
    )
    trainer.extend(log_report_extension)

    trainer.extend(ChainerPruningExtension(trial, "validation/main/accuracy", (1, "epoch")))

    # Run!
    trainer.run(show_loop_exception_msg=False)

    # Set the user attributes such as loss and accuracy for train and validation sets
    log_last = log_report_extension.log[-1]
    for key, value in log_last.items():
        trial.set_user_attr(key, value)

    # Return the validation accuracy
    return log_report_extension.log[-1]["validation/main/accuracy"]


if __name__ == "__main__":
    # This verbosity change is just to simplify the script output.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
