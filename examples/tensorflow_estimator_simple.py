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

import numpy as np
import tensorflow as tf

import optuna

MODEL_DIR = tempfile.mkdtemp()
BATCH_SIZE = 128
TRAIN_STEPS = 1000


def create_network(trial, features):
    # We optimize the numbers of layers and their units.
    input_layer = tf.reshape(features['x'], [-1, 784])
    prev_layer = input_layer

    n_layers = trial.suggest_int('n_layers', 1, 3)
    for i in range(n_layers):
        n_units = trial.suggest_int('n_units_l{}'.format(i), 1, 128)
        prev_layer = tf.keras.layers.Dense(
            units=n_units, activation=tf.nn.relu)(prev_layer)

    logits = tf.keras.layers.Dense(units=10)(prev_layer)
    return logits


def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.

    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    if optimizer_name == 'Adam':
        adam_lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=adam_lr)
    else:
        sgd_lr = trial.suggest_loguniform('sgd_lr', 1e-5, 1e-1)
        sgd_momentum = trial.suggest_loguniform('sgd_momentum', 1e-5, 1e-1)
        optimizer = tf.compat.v1.train.MomentumOptimizer(
            learning_rate=sgd_lr, momentum=sgd_momentum)

    return optimizer


def model_fn(trial, features, labels, mode):
    logits = create_network(trial, features)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = create_optimizer(trial)
        train_op = optimizer.minimize(loss, tf.compat.v1.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.compat.v1.metrics.accuracy(labels=labels,
                                                  predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def objective(trial):
    (train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.mnist.load_data()

    train_data = train_data / np.float32(255)
    train_labels = train_labels.astype(np.int32)

    eval_data = eval_data / np.float32(255)
    eval_labels = eval_labels.astype(np.int32)

    model_dir = "{}/{}".format(MODEL_DIR, trial.number)
    mnist_classifier = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode: model_fn(trial, features, labels, mode),
        model_dir=model_dir)

    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, y=train_labels, batch_size=BATCH_SIZE, num_epochs=None, shuffle=True)

    mnist_classifier.train(input_fn=train_input_fn, steps=TRAIN_STEPS)

    eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    return float(eval_results['accuracy'])


def main(unused_argv):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    shutil.rmtree(MODEL_DIR)


if __name__ == "__main__":
    tf.compat.v1.app.run()
