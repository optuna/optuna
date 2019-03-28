"""
Optuna example that optimizes multi-layer perceptrons using MX-NET.

In this example, we optimize the validation accuracy of hand-written digit recognition using
MX-NET and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python mxnet_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --storage sqlite:///example.db`
    $ optuna study optimize mxnet_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db

"""

from __future__ import print_function


import mxnet as mx
import numpy as np


N_TRAIN_EXAMPLES = 3000
N_TEST_EXAMPLES = 1000
BATCHSIZE = 128
EPOCH = 10


def model_fn(trial):
    n_layers = trial.suggest_int('n_layers', 1, 3)

    data = mx.symbol.Variable('data')
    data = mx.sym.flatten(data=data)
    for i in range(n_layers):
        num_hidden = int(trial.suggest_loguniform('n_units_l{}'.format(i), 4, 128))
        data = mx.symbol.FullyConnected(data=data, num_hidden=num_hidden)
        data = mx.symbol.Activation(data=data, act_type="relu")

    data = mx.symbol.FullyConnected(data=data, num_hidden=10)
    mlp = mx.symbol.SoftmaxOutput(data=data, name="softmax")

    return mlp


def create_optimizer(trial):
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'MomentumSGD'])

    if optimizer_name == 'Adam':
        adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
        optimizer = mx.optimizer.Adam(learning_rate=adam_lr, wd=weight_decay)
    else:
        momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        optimizer = mx.optimizer.SGD(momentum=momentum_sgd_lr, wd=weight_decay)

    return optimizer


def objective(trial):
    # Model and optimizer
    mlp = model_fn(trial)
    optimizer = create_optimizer(trial)

    # Dataset
    mnist = mx.test_utils.get_mnist()
    rng = np.random.RandomState(0)
    permute_train = rng.permutation(len(mnist['train_data']))
    train = mx.io.NDArrayIter(
        data=mnist['train_data'][permute_train][:N_TRAIN_EXAMPLES],
        label=mnist['train_label'][permute_train][:N_TRAIN_EXAMPLES],
        batch_size=BATCHSIZE,
        shuffle=True)
    permute_test = rng.permutation(len(mnist['test_data']))
    val = mx.io.NDArrayIter(
        data=mnist['test_data'][permute_test][:N_TEST_EXAMPLES],
        label=mnist['test_label'][permute_test][:N_TEST_EXAMPLES],
        batch_size=BATCHSIZE)

    # Callback
    def _callback(param):
        """The checkpoint function."""
        if param.eval_metric is not None:
            if param.epoch == 0 and param.nbatch == 1:
                print("%-12s %-12s" % ("epoch", "val-accuracy"))
            name_value = param.eval_metric.get_name_value()
            if param.nbatch == 1:
                for name, value in name_value:
                    print('%-12d %-12f' % (param.epoch, value))

    # Trainer
    model = mx.model.FeedForward(
        symbol=mlp,
        optimizer=optimizer,
        num_epoch=EPOCH,
    )
    model.fit(X=train, eval_data=val, batch_end_callback=_callback)

    # Return the 1.0 - accuracy
    test = mx.io.NDArrayIter(
        data=mnist['test_data'],
        label=mnist['test_label'],
        batch_size=BATCHSIZE)
    accuracy = model.score(X=test)

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
