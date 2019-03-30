"""
Optuna example that demonstrates a pruner for Keras.

In this example, we optimize the validation accuracy of hand-written digit recognition using
Keras and MNIST, where the architecture of the neural network is optimized. Throughout the
training of neural networks, a pruner observes intermediate results and stops unpromising trials.

You can run this example as follows:
    $ python keras_integration.py

"""

import optuna
from optuna.integration import KerasPruningCallback

import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers
from keras import regularizers


BATCHSIZE = 128
CLASSES = 10
EPOCHS = 20


def model_fn(trial):
    # Model
    n_layers = trial.suggest_int('n_layers', 1, 3)
    weight_decay = trial.suggest_uniform('weight_decay', 1e-10, 1e-3)
    model = Sequential()
    for i in range(n_layers):
        num_hidden = int(trial.suggest_loguniform('n_units_l{}'.format(i), 4, 128))
        model.add(Dense(num_hidden,
                        activation='relu',
                        kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Dense(CLASSES,
                    activation='softmax',
                    kernel_regularizer=regularizers.l2(weight_decay)))

    # Optimizer
    lr = trial.suggest_uniform('lr', 1e-5, 1e-1)
    optimizer = trial.suggest_categorical('optimizer', ['RMSprop', 'Adam', 'SGD'])
    model.compile(loss='categorical_crossentropy',
                  optimizer=getattr(optimizers, optimizer)(lr=lr),
                  metrics=['accuracy'])

    return model


def objective(trial):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32')/255
    x_test = x_test.reshape(10000, 784).astype('float32')/255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, CLASSES)
    y_test = keras.utils.to_categorical(y_test, CLASSES)

    # build model
    model = model_fn(trial)

    model.fit(x_train, y_train,
              batch_size=BATCHSIZE,
              callbacks=[KerasPruningCallback(trial, 'acc')],
              epochs=EPOCHS,
              validation_data=(x_test, y_test),
              verbose=1)

    score = model.evaluate(x_test, y_test, verbose=0)
    return score[1]


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=100)
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
