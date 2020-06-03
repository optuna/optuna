import os

from keras.backend import clear_session
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import mlflow
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import optuna

TEST_SIZE = 0.25
BATCHSIZE = 16
EPOCHS = 100


def standardize(data):
    return StandardScaler().fit_transform(data)


def create_model(num_features, trial):
    model = Sequential()
    model.add(
        Dense(
            num_features,
            activation="relu",
            kernel_initializer="normal",
            input_shape=(num_features,),
        )
    ),
    model.add(Dense(16, activation="relu", kernel_initializer="normal"))
    model.add(Dense(16, activation="relu", kernel_initializer="normal"))
    model.add(Dense(1, kernel_initializer="normal", activation="linear"))

    optimizer = SGD(
        lr=trial.suggest_loguniform("lr", 1e-5, 1e-1),
        momentum=trial.suggest_uniform("momentum", 0.0, 1.0),
    )
    model.compile(loss="mean_squared_error", optimizer=optimizer)
    return model


def mlflow_callback(study, trial):
    trial_value = trial.value if trial.value is not None else float("nan")
    with mlflow.start_run(run_name=study.study_name):
        mlflow.log_params(trial.params)
        mlflow.log_metrics({"mean_squared_error": trial_value})


def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    X, y = load_wine(return_X_y=True)
    X = standardize(X)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    model = create_model(X.shape[1], trial)
    model.fit(X_train, y_train, shuffle=True, batch_size=BATCHSIZE, epochs=EPOCHS, verbose=False)

    return model.evaluate(X_valid, y_valid, verbose=0)


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(
        objective,
        study_name="k8s_mlflow",
        n_trials=100,
        timeout=600,
        storage="postgresql://{}:{}@postgres:5432/{}".format(
            os.environ["POSTGRES_USER"],
            os.environ["POSTGRES_PASSWORD"],
            os.environ["POSTGRES_DB"],
        ),
        load_if_exists=True,
        callbacks=[optuna.integration.mlflow.MLflowCalback()]
    )

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
