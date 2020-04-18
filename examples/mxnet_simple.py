"""
Optuna example that optimizes multi-layer perceptrons using MXNet.

In this example, we optimize the validation accuracy of hand-written digit recognition using
MXNet and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python mxnet_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize mxnet_simple.py objective --n-trials=100 --study-name $STUDY_NAME \
      --storage sqlite:///example.db

"""

import logging

import mxnet as mx
import numpy as np

import optuna


N_TRAIN_EXAMPLES = 3000
N_TEST_EXAMPLES = 1000
BATCHSIZE = 128
EPOCH = 10

# Set log level for MXNet.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def create_model(trial):
    # We optimize the number of layers and hidden units in each layer.
    n_layers = trial.suggest_int("n_layers", 1, 3)

    data = mx.symbol.Variable("data")
    data = mx.sym.flatten(data=data)
    for i in range(n_layers):
        num_hidden = int(trial.suggest_loguniform("n_units_l{}".format(i), 4, 128))
        data = mx.symbol.FullyConnected(data=data, num_hidden=num_hidden)
        data = mx.symbol.Activation(data=data, act_type="relu")

    data = mx.symbol.FullyConnected(data=data, num_hidden=10)
    mlp = mx.symbol.SoftmaxOutput(data=data, name="softmax")

    return mlp


def create_optimizer(trial):
    # We optimize over the type of optimizer to use (Adam or SGD with momentum).
    # We also optimize over the learning rate and weight decay of the selected optimizer.
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-10, 1e-3)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "MomentumSGD"])

    if optimizer_name == "Adam":
        adam_lr = trial.suggest_loguniform("adam_lr", 1e-5, 1e-1)
        optimizer = mx.optimizer.Adam(learning_rate=adam_lr, wd=weight_decay)
    else:
        momentum_sgd_lr = trial.suggest_loguniform("momentum_sgd_lr", 1e-5, 1e-1)
        optimizer = mx.optimizer.SGD(momentum=momentum_sgd_lr, wd=weight_decay)

    return optimizer


def objective(trial):
    # Generate trial model and trial optimizer.
    mlp = create_model(trial)
    optimizer = create_optimizer(trial)

    # Load the test and train MNIST dataset.
    mnist = mx.test_utils.get_mnist()
    rng = np.random.RandomState(0)
    permute_train = rng.permutation(len(mnist["train_data"]))
    train = mx.io.NDArrayIter(
        data=mnist["train_data"][permute_train][:N_TRAIN_EXAMPLES],
        label=mnist["train_label"][permute_train][:N_TRAIN_EXAMPLES],
        batch_size=BATCHSIZE,
        shuffle=True,
    )
    permute_test = rng.permutation(len(mnist["test_data"]))
    val = mx.io.NDArrayIter(
        data=mnist["test_data"][permute_test][:N_TEST_EXAMPLES],
        label=mnist["test_label"][permute_test][:N_TEST_EXAMPLES],
        batch_size=BATCHSIZE,
    )

    # Create our MXNet trainable model and fit it on MNIST data.
    model = mx.mod.Module(symbol=mlp)
    model.fit(
        train_data=train,
        eval_data=val,
        optimizer=optimizer,
        optimizer_params={"rescale_grad": 1.0 / BATCHSIZE},
        num_epoch=EPOCH,
    )

    # Compute the accuracy on the entire test set.
    test = mx.io.NDArrayIter(
        data=mnist["test_data"], label=mnist["test_label"], batch_size=BATCHSIZE
    )
    accuracy = model.score(eval_data=test, eval_metric="acc")[0]

    return accuracy[1]


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
