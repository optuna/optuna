"""
Optuna example that demonstrates a pruner for tf.keras.

In this example, we optimize the validation accuracy of hand-written digit recognition
using tf.keras and MNIST, where the architecture of the neural network
and the parameters of optimizer are optimized.
Throughout the training of neural networks,
a pruner observes intermediate results and stops unpromising trials.

You can run this example as follows:
    $ python tfkeras_integration.py

"""

import tensorflow as tf
import tensorflow_datasets as tfds

import optuna
from optuna.integration import TFKerasPruningCallback


BATCHSIZE = 128
CLASSES = 10
EPOCHS = 20
N_TRAIN_EXAMPLES = 3000
STEPS_PER_EPOCH = int(N_TRAIN_EXAMPLES / BATCHSIZE / 10)
VALIDATION_STEPS = 30


def train_dataset():

    ds = tfds.load("mnist", split=tfds.Split.TRAIN, shuffle_files=True)
    ds = ds.map(lambda x: (tf.cast(x["image"], tf.float32) / 255.0, x["label"]))
    ds = ds.repeat().shuffle(1024).batch(BATCHSIZE)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def eval_dataset():

    ds = tfds.load("mnist", split=tfds.Split.TEST, shuffle_files=False)
    ds = ds.map(lambda x: (tf.cast(x["image"], tf.float32) / 255.0, x["label"]))
    ds = ds.repeat().batch(BATCHSIZE)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def create_model(trial):

    # Hyperparameters to be tuned by Optuna.
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.0, 1.0)
    units = trial.suggest_categorical("units", [32, 64, 128, 256, 512])

    # Compose neural network with one hidden layer.
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=units, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(CLASSES, activation=tf.nn.softmax))

    # Compile model.
    model.compile(
        optimizer=tf.keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=True),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def objective(trial):
    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()

    # Metrics to be monitored by Optuna.
    if tf.__version__ >= "2":
        monitor = "val_accuracy"
    else:
        monitor = "val_acc"

    # Create tf.keras model instance.
    model = create_model(trial)

    # Create dataset instance.
    ds_train = train_dataset()
    ds_eval = eval_dataset()

    # Create callbacks for early stopping and pruning.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3),
        TFKerasPruningCallback(trial, monitor),
    ]

    # Train model.
    history = model.fit(
        ds_train,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=ds_eval,
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks,
    )

    return history.history[monitor][-1]


def show_result(study):

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


def main():

    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2)
    )

    study.optimize(objective, n_trials=25, timeout=600)

    show_result(study)


if __name__ == "__main__":
    main()
