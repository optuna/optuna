"""
Optuna example that optimizes a neural network regressor for the
wine quality dataset using Keras and records hyperparamters and metrics using MLflow.

In this example, we optimize the learning rate and momentum of
stochastic gradient descent optimizer to minimize the validation mean squared error
for the wine quality regression.

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python keras_mlflow.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction minimize --storage sqlite:///example.db`
    $ optuna study optimize keras_mlflow.py objective --n-trials=100 \
      --study $STUDY_NAME --storage sqlite:///example.db


After the script finishes, run the MLflow UI:
    $ mlflow ui

and view the optimization results at http://127.0.0.1:5000.
"""

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
    model.add(Dense(num_features,
                    activation='relu',
                    kernel_initializer='normal',
                    input_shape=(num_features,))),
    model.add(Dense(16,
                    activation='relu',
                    kernel_initializer='normal'))
    model.add(Dense(16,
                    activation='relu',
                    kernel_initializer='normal'))
    model.add(Dense(1,
                    kernel_initializer='normal',
                    activation='linear'))

    optimizer = SGD(lr=trial.suggest_loguniform('lr', 1e-5, 1e-1),
                    momentum=trial.suggest_uniform('momentum', 0.0, 1.0))
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer)
    return model


def mlflow_callback(study, trial):
    with mlflow.start_run(run_name=study.study_name):
        mlflow.log_params(trial.params)
        mlflow.log_metrics({'mean_squared_error': trial.value})


def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    X, y = load_wine(return_X_y=True)
    X = standardize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    model = create_model(X.shape[1], trial)
    model.fit(X_train,
              y_train,
              shuffle=True,
              batch_size=BATCHSIZE,
              epochs=EPOCHS,
              verbose=False)

    return model.evaluate(X_test, y_test, verbose=0)


if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100, timeout=600, callbacks=[mlflow_callback])

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
