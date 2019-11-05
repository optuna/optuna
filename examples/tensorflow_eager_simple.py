"""
Optuna example that optimizes multi-layer perceptrons using Tensorflow (Eager Execution).

In this example, we optimize the validation accuracy of hand-written digit recognition using
Tensorflow and MNIST. We optimize the neural network architecture as well as the optimizer
configuration.

We have the following two ways to execute this example:

(1) Execute this code directly.
    $ python tensorflow_eager_simple.py


(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --direction maximize --storage sqlite:///example.db`
    $ optuna study optimize tensorflow_eager_simple.py objective --n-trials=100 \
      --study $STUDY_NAME --storage sqlite:///example.db

"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist

N_TRAIN_EXAMPLES = 3000
N_TEST_EXAMPLES = 1000
BATCHSIZE = 128
CLASSES = 10
EPOCHS = 1
tf.enable_eager_execution()


def create_model(trial):
    # We optimize the numbers of layers, their units and weight decay parameter.
    n_layers = trial.suggest_int('n_layers', 1, 3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    for i in range(n_layers):
        num_hidden = int(trial.suggest_loguniform('n_units_l{}'.format(i), 4, 128))
        model.add(
            tf.keras.layers.Dense(num_hidden,
                                  activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(
        tf.keras.layers.Dense(CLASSES, kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    return model


def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = ['RMSPropOptimizer', 'AdamOptimizer', 'MomentumOptimizer']
    optimizer_selected = trial.suggest_categorical('optimizer', optimizer_options)
    if optimizer_selected == 'RMSPropOptimizer':
        kwargs['learning_rate'] = trial.suggest_loguniform('rmsprop_learning_rate', 1e-5, 1e-1)
        kwargs['decay'] = trial.suggest_uniform('rmsprop_decay', 0.85, 0.99)
        kwargs['momentum'] = trial.suggest_loguniform('rmsprop_momentum', 1e-5, 1e-1)
    elif optimizer_selected == 'AdamOptimizer':
        kwargs['learning_rate'] = trial.suggest_loguniform('adam_learning_rate', 1e-5, 1e-1)
    elif optimizer_selected == 'MomentumOptimizer':
        kwargs['learning_rate'] = trial.suggest_loguniform('momentum_opt_learning_rate', 1e-5,
                                                           1e-1)
        kwargs['momentum'] = trial.suggest_loguniform('momentum_opt_momentum', 1e-5, 1e-1)

    optimizer = getattr(tf.train, optimizer_selected)(**kwargs)
    return optimizer


def learn(model, optimizer, dataset, mode='eval'):
    accuracy = tf.contrib.eager.metrics.Accuracy('accuracy', dtype=tf.float32)

    for batch, (images, labels) in enumerate(dataset):
        with tf.GradientTape() as tape:
            logits = model(images, training=(mode == 'train'))
            loss_value = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
            if mode == 'eval':
                accuracy(tf.argmax(logits, axis=1, output_type=tf.int64),
                         tf.cast(labels, tf.int64))
            else:
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables))

    if mode == 'eval':
        return accuracy


def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(60000).batch(BATCHSIZE).take(N_TRAIN_EXAMPLES)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.shuffle(10000).batch(BATCHSIZE).take(N_TEST_EXAMPLES)
    return train_ds, test_ds


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    # Get MNIST data.
    train_ds, test_ds = get_mnist()

    # Build model and optimizer.
    model = create_model(trial)
    optimizer = create_optimizer(trial)

    # Training and validating cycle.
    with tf.device("/cpu:0"):
        for _ in range(EPOCHS):
            learn(model, optimizer, train_ds, 'train')

        accuracy = learn(model, optimizer, test_ds, 'eval')

    # Return last validation accuracy.
    return accuracy.result()


if __name__ == '__main__':
    import optuna

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
