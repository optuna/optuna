"""
Optuna example that optimizes convolutional neural network and data augmentation using fastai V2.

In this example, we optimize the hyperparameters of a convolutional neural network and
data augmentation for hand-written digit recognition in terms of validation accuracy.
The network is implemented by fastai and
evaluated on MNIST dataset. Throughout the training of neural networks, a pruner observes
intermediate results and stops unpromising trials.
Note that this example will take longer than the other examples
as this uses the entire MNIST dataset.

You can run this example as follows, pruning can be turned on and off with the `--pruning`
argument.
    $ python fastaiv2_simple.py [--pruning]
"""

import argparse

from fastai.vision.all import accuracy
from fastai.vision.all import aug_transforms
from fastai.vision.all import CudaCallback
from fastai.vision.all import ImageDataLoaders
from fastai.vision.all import Learner
from fastai.vision.all import SimpleCNN
from fastai.vision.all import untar_data
from fastai.vision.all import URLs

import optuna
from optuna.integration import FastAIPruningCallback


BATCHSIZE = 128
EPOCHS = 10


path = untar_data(URLs.MNIST_SAMPLE)


def objective(trial):
    # Data Augmentation
    apply_tfms = trial.suggest_categorical("apply_tfms", [True, False])
    if apply_tfms:
        # MNIST is a hand-written digit dataset. Thus horizontal and vertical flipping are
        # disabled. However, the two flipping will be important when the dataset is CIFAR or
        # ImageNet.
        tfms = aug_transforms(
            do_flip=False,
            flip_vert=False,
            max_rotate=trial.suggest_int("max_rotate", 0, 45),
            max_zoom=trial.suggest_float("max_zoom", 1, 2),
            p_affine=trial.suggest_discrete_uniform("p_affine", 0.1, 1.0, 0.1),
        )
    data = ImageDataLoaders.from_folder(
        path, bs=BATCHSIZE, batch_tfms=tfms if apply_tfms else None
    )

    n_layers = trial.suggest_int("n_layers", 2, 5)

    n_channels = [3]
    for i in range(n_layers):
        out_channels = trial.suggest_int("n_channels_{}".format(i), 3, 32)
        n_channels.append(out_channels)
    n_channels.append(2)

    model = SimpleCNN(n_channels)

    learn = Learner(
        data,
        model,
        metrics=[accuracy],
        # You could as FastAIPruningCallback in the fit function
        cbs=[FastAIPruningCallback(trial), CudaCallback],
    )

    # See https://forums.fast.ai/t/how-to-diable-progress-bar-completely/65249/3
    # to disable progress bar and logging info
    with learn.no_bar():
        with learn.no_logging():
            learn.fit(EPOCHS)

    return learn.validate()[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fastai V2 example.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
