"""
Optuna example that optimizes multi-layer perceptrons using Tensorflow (Estimator API).

In this example, we optimize the validation accuracy of hand-written digit recognition using
Tensorflow and MNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole MNIST dataset, we here use a small
subset of it.

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python tensorflow_estimator_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize tensorflow_estimator_simple.py objective --n-trials=100 \
      --study $STUDY_NAME --storage sqlite:///example.db

"""

import shutil
import tempfile

import tensorflow as tf
import tensorflow_datasets as tfds

import optuna

MODEL_DIR = tempfile.mkdtemp()
BATCH_SIZE = 128
TRAIN_STEPS = 1000
N_TRAIN_BATCHES = 3000
N_TEST_BATCHES = 1000


def preprocess(image, label):
    image = tf.reshape(image, [-1, 28 * 28])
    image = tf.cast(image, tf.float32)
    image /= 255
    label = tf.cast(label, tf.int32)
    return {"x": image}, label


def train_input_fn():
    data = tfds.load(name="mnist", as_supervised=True)
    train_ds = data["train"]
    train_ds = train_ds.map(preprocess).shuffle(60000).batch(BATCH_SIZE).take(N_TRAIN_BATCHES)
    return train_ds


def eval_input_fn():
    data = tfds.load(name="mnist", as_supervised=True)
    test_ds = data["test"]
    test_ds = test_ds.map(preprocess).shuffle(10000).batch(BATCH_SIZE).take(N_TEST_BATCHES)
    return test_ds


def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    if optimizer_name == "Adam":
        adam_lr = trial.suggest_loguniform("adam_lr", 1e-5, 1e-1)
        return lambda: tf.keras.optimizers.Adam(learning_rate=adam_lr)
    else:
        sgd_lr = trial.suggest_loguniform("sgd_lr", 1e-5, 1e-1)
        sgd_momentum = trial.suggest_loguniform("sgd_momentum", 1e-5, 1e-1)
        return lambda: tf.keras.optimizers.SGD(learning_rate=sgd_lr, momentum=sgd_momentum)


def create_classifier(trial):
    # We optimize the numbers of layers and their units.

    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_units = []
    for i in range(n_layers):
        n_units = trial.suggest_int("n_units_l{}".format(i), 1, 128)
        hidden_units.append(n_units)

    optimizer = create_optimizer(trial)

    model_dir = "{}/{}".format(MODEL_DIR, trial.number)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=[tf.feature_column.numeric_column("x", shape=[28 * 28])],
        hidden_units=hidden_units,
        model_dir=model_dir,
        n_classes=10,
        optimizer=optimizer,
    )

    return classifier


def objective(trial):
    classifier = create_classifier(trial)

    classifier.train(input_fn=train_input_fn, steps=TRAIN_STEPS)

    eval_results = classifier.evaluate(input_fn=eval_input_fn)

    return float(eval_results["accuracy"])


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    shutil.rmtree(MODEL_DIR)


if __name__ == "__main__":
    main()
