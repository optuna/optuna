"""
Optuna example that demonstrates a pruner for fastai.

In this example, we optimize the hyperparameters of a convolutional neural network for hand-written
digit recognition in terms of validation accuracy. The network is implemented by fastai and
evaluated on MNIST dataset. Throughout the training of neural networks, a pruner observes
intermediate results and stops unpromising trials.
Note that this example will take longer than the other examples
as this uses the entire MNIST dataset.

You can run this example as follows, pruning can be turned on and off with the `--pruning`
argument.
    $ python fastai_integration.py [--pruning]
"""

import argparse
from functools import partial

from fastai import vision

import optuna
from optuna.integration import FastAIPruningCallback


BATCHSIZE = 128
EPOCHS = 10


path = vision.untar_data(vision.URLs.MNIST_SAMPLE)
data = vision.ImageDataBunch.from_folder(path, bs=BATCHSIZE)


def objective(trial):
    n_layers = trial.suggest_int('n_layers', 2, 5)

    n_channels = [3]
    for i in range(n_layers):
        out_channels = trial.suggest_int('n_channels_{}'.format(i), 3, 32)
        n_channels.append(out_channels)
    n_channels.append(2)

    model = vision.simple_cnn(n_channels)

    learn = vision.Learner(
        data, model, silent=True, metrics=[vision.accuracy],
        callback_fns=[partial(FastAIPruningCallback, trial=trial, monitor='valid_loss')])
    learn.fit(EPOCHS)

    return learn.validate()[-1].item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FastAI example.')
    parser.add_argument(
        '--pruning', '-p', action='store_true',
        help='Activate the pruning feature. `MedianPruner` stops unpromising '
             'trials at the early stages of training.')
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
