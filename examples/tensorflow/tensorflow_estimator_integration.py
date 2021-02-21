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

import tensorflow as tf
import tensorflow_datasets as tfds

import optuna


MODEL_DIR = tempfile.mkdtemp()
BATCH_SIZE = 128
TRAIN_STEPS = 1000
PRUNING_INTERVAL_STEPS = 50
N_TRAIN_BATCHES = 3000
N_VALID_BATCHES = 1000


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
    valid_ds = data["test"]
    valid_ds = valid_ds.map(preprocess).shuffle(10000).batch(BATCH_SIZE).take(N_VALID_BATCHES)
    return valid_ds


def create_classifier(trial):
    # We optimize the numbers of layers and their units.

    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_units = []
    for i in range(n_layers):
        n_units = trial.suggest_int("n_units_l{}".format(i), 1, 128)
        hidden_units.append(n_units)

    config = tf.estimator.RunConfig(
        save_summary_steps=PRUNING_INTERVAL_STEPS, save_checkpoints_steps=PRUNING_INTERVAL_STEPS
    )

    model_dir = "{}/{}".format(MODEL_DIR, trial.number)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=[tf.feature_column.numeric_column("x", shape=[28 * 28])],
        hidden_units=hidden_units,
        model_dir=model_dir,
        n_classes=10,
        optimizer=lambda: tf.keras.optimizers.Adam(learning_rate=0.01),
        config=config,
    )

    return classifier


def objective(trial):
    classifier = create_classifier(trial)

    optuna_pruning_hook = optuna.integration.TensorFlowPruningHook(
        trial=trial,
        estimator=classifier,
        metric="accuracy",
        run_every_steps=PRUNING_INTERVAL_STEPS,
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=TRAIN_STEPS, hooks=[optuna_pruning_hook]
    )

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, start_delay_secs=0, throttle_secs=0)

    eval_results, _ = tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    return float(eval_results["accuracy"])


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)
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

    shutil.rmtree(MODEL_DIR)


if __name__ == "__main__":
    main()
