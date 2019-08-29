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

from keras.backend import clear_session
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop

import optuna


N_TRAIN_EXAMPLES = 3000
N_TEST_EXAMPLES = 1000
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 10


def objective(trial):
    # Clear clutter from previous Keras session graphs.
    clear_session()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_x, img_y = x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(-1, img_x, img_y, 1)[:N_TRAIN_EXAMPLES].astype('float32') / 255
    x_test = x_test.reshape(-1, img_x, img_y, 1)[:N_TEST_EXAMPLES].astype('float32') / 255
    y_train = y_train[:N_TRAIN_EXAMPLES]
    y_test = y_train[:N_TEST_EXAMPLES]
    input_shape = (img_x, img_y, 1)

    model = Sequential()
    model.add(
        Conv2D(filters=trial.suggest_categorical('filters', [32, 64]),
               kernel_size=trial.suggest_categorical('kernel_size', [3, 5]),
               strides=trial.suggest_categorical('strides', [1, 2]),
               activation=trial.suggest_categorical('activation', ['relu', 'linear']),
               input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(CLASSES, activation='softmax'))

    # We compile our model with a sampled learning rate.
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=RMSprop(lr=lr),
                  metrics=['accuracy'])

    model.fit(x_train,
              y_train,
              validation_data=(x_test, y_test),
              shuffle=True,
              batch_size=BATCHSIZE,
              epochs=EPOCHS,
              verbose=False)

    # Evaluate the model accuracy on the test set.
    score = model.evaluate(x_test, y_test, verbose=0)
    return score[1]


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
