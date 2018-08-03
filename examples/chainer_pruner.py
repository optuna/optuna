"""
PFNOpt example that demonstrates a pruner for Chainer.

In this example, we optimize the validation accuracy of hand-written digit recognition using
Chainer and MNIST. We optimize the neural network architecture. Throughout training of
neural networks, a pruner observes intermediate results and stops unpromising trials.

You can run this example as follows:
    $ python chainer_pruner.py

"""

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import pkg_resources

import pfnopt


if pkg_resources.parse_version(chainer.__version__) < pkg_resources.parse_version('4.0.0'):
    raise RuntimeError('Chainer>=4.0.0 is required for this example.')


N_TRAIN_EXAMPLES = 3000
N_TEST_EXAMPLES = 1000
BATCHSIZE = 128
EPOCH = 10
PRUNER_INTERVAL = 3


def create_model(trial):
    # We optimize the numbers of layers and their units.
    n_layers = trial.suggest_int('n_layers', 1, 3)

    layers = []
    for i in range(n_layers):
        n_units = int(trial.suggest_loguniform('n_units_l{}'.format(i), 32, 256))
        layers.append(L.Linear(None, n_units))
        layers.append(F.relu)
    layers.append(L.Linear(None, 10))

    return chainer.Sequential(*layers)


def objective(trial):
    # Model and optimizer
    model = L.Classifier(create_model(trial))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

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

    # Add an extension for Chainer pruner.
    trainer.extend(
        pfnopt.integration.ChainerPruningExtension(trial, 'validation/main/loss',
                                                   (PRUNER_INTERVAL, 'epoch'))
    )

    trainer.extend(chainer.training.extensions.Evaluator(test_iter, model))
    trainer.extend(chainer.training.extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    log_report_extension = chainer.training.extensions.LogReport(log_name=None)
    trainer.extend(log_report_extension)

    # Run!
    trainer.run(show_loop_exception_msg=False)

    # Return the validation error.
    val_err = 1.0 - log_report_extension.log[-1]['validation/main/accuracy']
    return val_err


if __name__ == '__main__':
    study = pfnopt.minimize(objective, n_trials=100, pruner=pfnopt.pruners.MedianPruner())
    pruned_trials = [t for t in study.trials if t.state == pfnopt.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == pfnopt.structs.TrialState.COMPLETE]
    print('Result: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))
    print('  Best value: ', study.best_trial.value)
