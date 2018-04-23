"""
PFNOpt example that optimizes multi-layer perceptrons using Chainer.

In this example, we optimize the validation accuracy of hand-written digit recognition using
Chainer and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python chainer_mnist.py


(2) Execute through CLI.
    $ pfnopt minimize chainer_mnist.py objective --create-study --n-trials=100

"""

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


N_TRAIN_EXAMPLES = 3000
N_TEST_EXAMPLES = 1000
BATCHSIZE = 128
EPOCH = 10


def create_model(client):
    # We optimize the numbers of layers and their units.
    n_layers = int(client.sample_uniform('n_layers', 1, 4))

    layers = []
    for i in range(n_layers):
        n_units = int(client.sample_loguniform('n_units_l{}'.format(i), 4, 128))
        layers.append(L.Linear(None, n_units))
        layers.append(F.relu)
    layers.append(L.Linear(None, 10))

    return chainer.Sequential(*layers)


def create_optimizer(client, model):
    # We optimize the choice of optimizers as well as their learning rates.
    optimizer_name = client.sample_categorical('optimizer', ['Adam', 'MomentumSGD'])
    if optimizer_name == 'Adam':
        adam_alpha = client.sample_loguniform('adam_apha', 1e-5, 1e-1)
        optimizer = chainer.optimizers.Adam(alpha=adam_alpha)
    else:
        momentum_sgd_lr = client.sample_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        optimizer = chainer.optimizers.MomentumSGD(lr=momentum_sgd_lr)

    weight_decay = client.sample_loguniform('weight_decay', 1e-10, 1e-3)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    return optimizer


def objective(client):
    # Model and optimizer
    model = L.Classifier(create_model(client))
    optimizer = create_optimizer(client, model)

    # Dataset
    rng = np.random.RandomState(0)
    train, test = chainer.datasets.get_mnist()
    train = chainer.datasets.SubDataset(
        train, 0, N_TRAIN_EXAMPLES, order=rng.permutation(len(train)))
    test = chainer.datasets.SubDataset(
        test, 0, N_TEST_EXAMPLES, order=rng.permutation(len(test)))
    train_iter = chainer.iterators.SerialIterator(train, BATCHSIZE)
    test_iter = chainer.iterators.SerialIterator(test, BATCHSIZE, repeat=False, shuffle=False)

    # Trainer
    updater = chainer.training.StandardUpdater(train_iter, optimizer)
    trainer = chainer.training.Trainer(updater, (EPOCH, 'epoch'))
    trainer.extend(chainer.training.extensions.Evaluator(test_iter, model))
    log_report_extension = chainer.training.extensions.LogReport(log_name=None)
    trainer.extend(chainer.training.extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(log_report_extension)

    # Run!
    trainer.run()

    # Set the user attributes such as loss and accuracy for train and validation sets
    log_last = log_report_extension.log[-1]
    for key, value in log_last.items():
        client.set_user_attr(key, value)

    # Return the validation error
    val_err = 1.0 - log_report_extension.log[-1]['validation/main/accuracy']
    return val_err


if __name__ == '__main__':
    import pfnopt
    study = pfnopt.minimize(objective, n_trials=100)
    print(study.best_trial)
