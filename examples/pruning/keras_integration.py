"""
Optuna example that demonstrates a pruner for Keras.

In this example, we optimize the validation accuracy of hand-written digit recognition using
Keras and MNIST, where the architecture of the neural network and the learning rate of optimizer
is optimized. Throughout the training of neural networks, a pruner observes intermediate
results and stops unpromising trials.

You can run this example as follows:
    $ python keras_integration.py

"""

import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential


import optuna
from optuna.integration import KerasPruningCallback


N_TRAIN_EXAMPLES = 3000
N_TEST_EXAMPLES = 1000
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 20


def create_model(trial):
    # We optimize the number of layers, hidden units and dropout in each layer and
    # the learning rate of RMSProp optimizer.

    # We define our MLP.
    n_layers = trial.suggest_int('n_layers', 1, 3)
    model = Sequential()
    for i in range(n_layers):
        num_hidden = int(trial.suggest_loguniform('n_units_l{}'.format(i), 4, 128))
        model.add(Dense(num_hidden,
                        activation='relu'))
        dropout = trial.suggest_uniform('dropout_l{}'.format(i), 0.2, 0.5)
        model.add(Dropout(rate=dropout))
    model.add(Dense(CLASSES, activation='softmax'))

    # We compile our model with a sampled learning rate.
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(lr=lr),
                  metrics=['accuracy'])

    return model


def objective(trial):
    # Clear clutter form previous session graphs.
    keras.backend.clear_session()

    # The data is split between train and test sets.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)[:N_TRAIN_EXAMPLES].astype('float32')/255
    x_test = x_test.reshape(10000, 784)[:N_TEST_EXAMPLES].astype('float32')/255

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train[:N_TRAIN_EXAMPLES], CLASSES)
    y_test = keras.utils.to_categorical(y_test[:N_TEST_EXAMPLES], CLASSES)

    # Generate our trial model.
    model = create_model(trial)

    # Fit the model on the training data.
    # The KerasPruningCallback checks for pruning condition every epoch.
    model.fit(x_train, y_train,
              batch_size=BATCHSIZE,
              callbacks=[KerasPruningCallback(trial, 'val_acc')],
              epochs=EPOCHS,
              validation_data=(x_test, y_test),
              verbose=1)

    # Evaluate the model accuracy on the test set.
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
