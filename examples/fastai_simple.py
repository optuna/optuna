"""
Optuna example that demonstrates a pruner for fastai.

In this example, we optimize the hyperparameters of a convolutional neural network for hand-written
digit recognition in terms of validation loss. The network is implemented by fastai and evaluated
on MNIST dataset. Throughout the training of neural networks, a pruner observes intermediate
results and stops unpromising trials.

You can run this example as follows:
    $ python fastai_integration.py

"""

from functools import partial

from fastai.vision import ImageDataBunch
from fastai.vision import Learner
from fastai.vision import simple_cnn
from fastai.vision import untar_data
from fastai.vision import URLs
import optuna
from optuna.integration import FastAIPruningCallback


path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)


def objective(trial: optuna.trial.Trial) -> float:
    n_layers = trial.suggest_int('n_layers', 2, 5)

    n_channels = [3]
    for i in range(n_layers):
        out_channels = trial.suggest_int('n_channels_{}'.format(i), 3, 32)
        n_channels.append(out_channels)
    n_channels.append(2)

    model = simple_cnn(n_channels)

    learn = Learner(
        data, model, silent=True,
        callback_fns=[partial(FastAIPruningCallback, trial=trial, monitor='valid_loss')])
    learn.fit(10)

    return learn.validate()[-1].item()


if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=100, timeout=600)
