import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet.gluon import nn
import numpy as np

import optuna


CUDA = False
EPOCHS = 10
BATCHSIZE = 128
LOG_INTERVAL = 100


def define_model(trial):
    net = nn.Sequential()
    n_layers = trial.suggest_int("n_layers", 1, 3)
    for i in range(n_layers):
        nodes = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        net.add(nn.Dense(nodes, activation="relu"))
    net.add(nn.Dense(10))
    return net


def transform(data, label):
    data = data.reshape((-1,)).astype(np.float32) / 255
    return data, label


def validate(ctx, val_data, net):
    metric = mx.metric.Accuracy()
    for data, label in val_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])

    return metric.get()


def objective(trial):
    if CUDA:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu()

    train_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST("./data", train=True).transform(transform),
        shuffle=True,
        batch_size=BATCHSIZE,
        last_batch="discard",
    )

    val_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST("./data", train=False).transform(transform),
        batch_size=BATCHSIZE,
        shuffle=False,
    )

    net = define_model(trial)

    # Collect all parameters from net and its children, then initialize them.
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    # Trainer is for updating parameters with gradient.
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    trainer = gluon.Trainer(net.collect_params(), optimizer_name, {"learning_rate": lr})
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    val_acc = 0

    for epoch in range(EPOCHS):
        # Reset data iterator and metric at beginning of epoch.
        metric.reset()
        for i, (data, label) in enumerate(train_data):
            # Copy data to ctx if necessary.
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Start recording computation graph with record() section.
            # Recorded graphs can then be differentiated with backward.
            with autograd.record():
                output = net(data)
                L = loss(output, label)
                L.backward()
            # Take a gradient step with batch_size equal to data.shape[0].
            trainer.step(data.shape[0])
            # Update metric at last.
            metric.update([label], [output])

            if i % LOG_INTERVAL == 0 and i > 0:
                name, acc = metric.get()
                print(f"[Epoch {epoch} Batch {i}] Training: {name}={acc}")

        name, acc = metric.get()
        print(f"[Epoch {epoch}] Training: {name}={acc}")

        name, val_acc = validate(ctx, val_data, net)
        print(f"[Epoch {epoch}] Validation: {name}={val_acc}")

        trial.report(val_acc, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    net.save_parameters("mnist.params")

    return val_acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
