"""
Optuna example that demonstrates a pruner for Tensorflow (Estimator API).

In this example, we optimize the hyperparameters of a neural network for hand-written
digit recognition in terms of validation accuracy. The network is implemented by Tensorflow and
evaluated by MNIST dataset. Throughout the training of neural networks, a pruner observes
intermediate results and stops unpromising trials.

You can run this example as follows:
    $ python tensorflow_estimator_integration.py

"""

import shutil
import tempfile

import numpy as np
import tensorflow as tf

import optuna

MODEL_DIR = tempfile.mkdtemp()
BATCH_SIZE = 128
TRAIN_STEPS = 1000
EVAL_STEPS = 100
PRUNING_INTERVAL_STEPS = 50


def create_network(trial, features):
    # We optimize the numbers of layers and their units.
    input_layer = tf.reshape(features['x'], [-1, 784])
    prev_layer = input_layer

    n_layers = trial.suggest_int('n_layers', 1, 3)
    for i in range(n_layers):
        n_units = trial.suggest_int('n_units_l{}'.format(i), 1, 128)
        prev_layer = tf.keras.layers.Dense(units=n_units, activation=tf.nn.relu)(prev_layer)

    logits = tf.keras.layers.Dense(units=10)(prev_layer)
    return logits


def model_fn(trial, features, labels, mode):
    logits = create_network(trial, features)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.compat.v2.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.compat.v2.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.compat.v2.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.compat.v2.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def objective(trial):
    (train_data, train_labels), (eval_data, eval_labels) = tf.keras.datasets.mnist.load_data()

    train_data = train_data / np.float32(255)
    train_labels = train_labels.astype(np.int32)

    eval_data = eval_data / np.float32(255)
    eval_labels = eval_labels.astype(np.int32)

    config = tf.estimator.RunConfig(
        save_summary_steps=PRUNING_INTERVAL_STEPS, save_checkpoints_steps=PRUNING_INTERVAL_STEPS)

    model_dir = "{}/{}".format(MODEL_DIR, trial.number)
    mnist_classifier = tf.estimator.Estimator(
        model_fn=lambda features, labels, mode: model_fn(trial, features, labels, mode),
        model_dir=model_dir,
        config=config)

    optuna_pruning_hook = optuna.integration.TensorFlowPruningHook(
        trial=trial,
        estimator=mnist_classifier,
        metric="accuracy",
        run_every_steps=PRUNING_INTERVAL_STEPS,
    )

    train_input_fn = tf.compat.v2.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, y=train_labels, batch_size=BATCH_SIZE, num_epochs=None, shuffle=True)

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=TRAIN_STEPS, hooks=[optuna_pruning_hook])

    eval_input_fn = tf.compat.v2.estimator.inputs.numpy_input_fn(
        x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn, steps=EVAL_STEPS, start_delay_secs=0, throttle_secs=0)

    eval_results, _ = tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)
    return float(eval_results['accuracy'])


def main(unused_argv):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25)
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

    shutil.rmtree(MODEL_DIR)


if __name__ == "__main__":
    tf.compat.v2.app.run()
