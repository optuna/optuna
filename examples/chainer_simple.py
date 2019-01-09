"""
Optuna example that optimizes multi-layer perceptrons using Chainer.

In this example, we optimize the validation accuracy of hand-written digit recognition using
Chainer and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python chainer_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --storage sqlite:///example.db`
    $ optuna study optimize chainer_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db

"""

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import pkg_resources

if pkg_resources.parse_version(chainer.__version__) < pkg_resources.parse_version('4.0.0'):
    raise RuntimeError('Chainer>=4.0.0 is required for this example.')


N_TRAIN_EXAMPLES = 3000
N_TEST_EXAMPLES = 1000
BATCHSIZE = 128
EPOCH = 10


def create_model(trial):
    # We optimize the numbers of layers and their units.
    n_layers = trial.suggest_int('n_layers', 1, 3)

    layers = []
    for i in range(n_layers):
        n_units = int(trial.suggest_loguniform('n_units_l{}'.format(i), 4, 128))
        layers.append(L.Linear(None, n_units))
        layers.append(F.relu)
    layers.append(L.Linear(None, 10))

    return chainer.Sequential(*layers)


def create_optimizer(trial, model):
    # We optimize the choice of optimizers as well as their parameters.
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'MomentumSGD'])
    if optimizer_name == 'Adam':
        adam_alpha = trial.suggest_loguniform('adam_alpha', 1e-5, 1e-1)
        optimizer = chainer.optimizers.Adam(alpha=adam_alpha)
    else:
        momentum_sgd_lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
        optimizer = chainer.optimizers.MomentumSGD(lr=momentum_sgd_lr)

    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
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
        trial.set_user_attr(key, value)

    # Return the validation error
    val_err = 1.0 - log_report_extension.log[-1]['validation/main/accuracy']
    return val_err


if __name__ == '__main__':
    import optuna
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    print('  User attrs:')
    for key, value in trial.user_attrs.items():
        print('    {}: {}'.format(key, value))
