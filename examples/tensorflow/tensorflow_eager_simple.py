"""
Optuna example that optimizes multi-layer perceptrons using Tensorflow (Eager Execution).

In this example, we optimize the validation accuracy of hand-written digit recognition using
Tensorflow and MNIST. We optimize the neural network architecture as well as the optimizer
configuration.

"""

from packaging import version
import tensorflow as tf
from tensorflow.keras.datasets import mnist

import optuna


if version.parse(tf.__version__) < version.parse("2.0.0"):
    raise RuntimeError("tensorflow>=2.0.0 is required for this example.")

N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 1


def create_model(trial):
    # We optimize the numbers of layers, their units and weight decay parameter.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-3, log=True)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for i in range(n_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        model.add(
            tf.keras.layers.Dense(
                num_hidden,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            )
        )
    model.add(
        tf.keras.layers.Dense(CLASSES, kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    )
    return model


def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["decay"] = trial.suggest_float("rmsprop_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer


def learn(model, optimizer, dataset, mode="eval"):
    accuracy = tf.metrics.Accuracy("accuracy", dtype=tf.float32)

    for batch, (images, labels) in enumerate(dataset):
        with tf.GradientTape() as tape:
            logits = model(images, training=(mode == "train"))
            loss_value = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            )
            if mode == "eval":
                accuracy(
                    tf.argmax(logits, axis=1, output_type=tf.int64), tf.cast(labels, tf.int64)
                )
            else:
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables))

    if mode == "eval":
        return accuracy


def get_mnist():
    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_valid = x_valid.astype("float32") / 255

    y_train = y_train.astype("int32")
    y_valid = y_valid.astype("int32")

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(60000).batch(BATCHSIZE).take(N_TRAIN_EXAMPLES)

    valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    valid_ds = valid_ds.shuffle(10000).batch(BATCHSIZE).take(N_VALID_EXAMPLES)
    return train_ds, valid_ds


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    # Get MNIST data.
    train_ds, valid_ds = get_mnist()

    # Build model and optimizer.
    model = create_model(trial)
    optimizer = create_optimizer(trial)

    # Training and validating cycle.
    with tf.device("/cpu:0"):
        for _ in range(EPOCHS):
            learn(model, optimizer, train_ds, "train")

        accuracy = learn(model, optimizer, valid_ds, "eval")

    # Return last validation accuracy.
    return accuracy.result()


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
