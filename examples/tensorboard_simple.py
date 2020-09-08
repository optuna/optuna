import tensorflow as tf

import optuna
from optuna.integration.tensorboard import TensorBoardCallback

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def train_test_model(num_units: int, dropout_rate: float, optimizer: str) -> float:
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_units, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        ]
    )
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(x_train, y_train, epochs=1)  # Run with 1 epoch to speed things up for demo purposes
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy


def objective(trial: optuna.trial.Trial) -> float:
    num_units = trial.suggest_int("NUM_UNITS", 16, 32)
    dropout_rate = trial.suggest_uniform("DROPOUT_RATE", 0.1, 0.2)
    optimizer = trial.suggest_categorical("OPTIMIZER", ["sgd", "adam"])

    accuracy = train_test_model(num_units, dropout_rate, optimizer)  # type: ignore
    return accuracy


tensorboard_callback = TensorBoardCallback("logs/", metric_name="accuracy")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10, timeout=600, callbacks=[tensorboard_callback])
