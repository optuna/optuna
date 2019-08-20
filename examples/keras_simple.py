"""
Optuna example that optimizes a neural network classifier configuration for the
MNIST dataset using Keras.

In this example, we optimize the validation accuracy of MNIST classification using
Keras. We optimize the filter and kernel size, kernel stride and layer activation.

We have following two ways to execute this example:

(1) Execute this code directly.
    $ python keras_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize keras_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db

"""

import numpy as np

from sklearn.metrics import accuracy_score

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam

import optuna


def objective(trial):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_x, img_y = x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(-1, img_x, img_y, 1)
    x_test = x_test.reshape(-1, img_x, img_y, 1)
    num_classes = 10
    input_shape = (img_x, img_y, 1)

    model = Sequential()
    model.add(
        Conv2D(filters=trial.suggest_categorical('filters', [32, 64]),
               kernel_size=trial.suggest_categorical('kernel_size', [3, 5]),
               strides=trial.suggest_categorical('strides', [1, 2]),
               activation=trial.suggest_categorical('activation', ['relu', 'linear']),
               input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train,
              y_train,
              validation_data=(x_test, y_test),
              shuffle=True,
              batch_size=512,
              epochs=2,
              verbose=False)

    preds = model.predict(x_test)
    pred_labels = np.argmax(preds, axis=1)
    accuracy = accuracy_score(y_test, pred_labels)
    return accuracy


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200, timeout=200)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
