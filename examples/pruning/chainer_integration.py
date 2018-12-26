"""
Optuna example that demonstrates a pruner for Chainer.

In this example, we optimize the hyperparameters of a neural network for hand-written
digit recognition in terms of validation loss. The network is implemented by Chainer and
evaluated by MNIST dataset. Throughout the training of neural networks, a pruner observes
intermediate results and stops unpromising trials.

You can run this example as follows:
    $ python chainer_integration.py

"""

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import pkg_resources

import optuna


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


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    model = L.Classifier(create_model(trial))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    rng = np.random.RandomState(0)
    train, test = chainer.datasets.get_mnist()
    train = chainer.datasets.SubDataset(
        train, 0, N_TRAIN_EXAMPLES, order=rng.permutation(len(train)))
    test = chainer.datasets.SubDataset(
        test, 0, N_TEST_EXAMPLES, order=rng.permutation(len(test)))
    train_iter = chainer.iterators.SerialIterator(train, BATCHSIZE)
    test_iter = chainer.iterators.SerialIterator(test, BATCHSIZE, repeat=False, shuffle=False)

    # Setup trainer.
    updater = chainer.training.StandardUpdater(train_iter, optimizer)
    trainer = chainer.training.Trainer(updater, (EPOCH, 'epoch'))

    # Add Chainer extension for pruners.
    trainer.extend(
        optuna.integration.ChainerPruningExtension(trial, 'validation/main/loss',
                                                   (PRUNER_INTERVAL, 'epoch'))
    )

    trainer.extend(chainer.training.extensions.Evaluator(test_iter, model))
    trainer.extend(chainer.training.extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    log_report_extension = chainer.training.extensions.LogReport(log_name=None)
    trainer.extend(log_report_extension)

    # Run training.
    # Please set show_loop_exception_msg False to inhibit messages about TrialPruned exception.
    # ChainerPruningExtension raises TrialPruned exception to stop training, and
    # trainer shows some messages every time it receive TrialPruned.
    trainer.run(show_loop_exception_msg=False)

    # Save loss and accuracy to user attributes.
    log_last = log_report_extension.log[-1]
    for key, value in log_last.items():
        trial.set_user_attr(key, value)

    return log_report_extension.log[-1]['validation/main/loss']


if __name__ == '__main__':
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)
    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
    print('Study statistics: ')
    print('  Number of finished trials: ', len(study.trials))
    print('  Number of pruned trials: ', len(pruned_trials))
    print('  Number of complete trials: ', len(complete_trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    print('  User attrs:')
    for key, value in trial.user_attrs.items():
        print('    {}: {}'.format(key, value))
