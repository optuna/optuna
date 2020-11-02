# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file
from __future__ import print_function

import argparse
import logging

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn

import optuna

# Parse CLI arguments

parser = argparse.ArgumentParser(description='MXNet Gluon MNIST Example')
parser.add_argument('--batch-size', type=int, default=100,
                    help='batch size for training and testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Train on GPU with CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
opt = parser.parse_args()


# define network

def network(trial):
    net = nn.Sequential()
    n_layers = trial.suggest_int("n_layers", 1, 3)
    for i in range(n_layers):
        nodes = trial.suggest_int("n_units_l{}".format(i), 4, 128)
        net.add(nn.Dense(nodes, activation='relu'))
    net.add(nn.Dense(10))
    return net

# data

def transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32)/255
    return data, label

# train

def test(ctx, val_data, net):
    metric = mx.metric.Accuracy()
    for data, label in val_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        metric.update([label], [output])

    return metric.get()


def objective(trial):
    if opt.cuda:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu()
    epochs = opt.epochs

    train_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST('./data', train=True).transform(transformer),
        batch_size=opt.batch_size, shuffle=True, last_batch='discard')

    val_data = gluon.data.DataLoader(
        gluon.data.vision.MNIST('./data', train=False).transform(transformer),
        batch_size=opt.batch_size, shuffle=False)

    net = network(trial)

    # Collect all parameters from net and its children, then initialize them.
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    optimizer_name = trial.suggest_categorical("optimizer",
                                               ["Adam", "RMSprop", "SGD"])
    # Trainer is for updating parameters with gradient.
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    trainer = gluon.Trainer(net.collect_params(), optimizer_name,
                            {'learning_rate': lr})
    metric = mx.metric.Accuracy()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(epochs):
        # reset data iterator and metric at begining of epoch.
        metric.reset()
        for i, (data, label) in enumerate(train_data):
            # Copy data to ctx if necessary
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Start recording computation graph with record() section.
            # Recorded graphs can then be differentiated with backward.
            with autograd.record():
                output = net(data)
                L = loss(output, label)
                L.backward()
            # take a gradient step with batch_size equal to data.shape[0]
            trainer.step(data.shape[0])
            # update metric at last.
            metric.update([label], [output])

            if i % opt.log_interval == 0 and i > 0:
                name, acc = metric.get()
                print('[Epoch %d Batch %d] Training: %s=%f'%(epoch, i, name, acc))

        name, acc = metric.get()
        print('[Epoch %d] Training: %s=%f'%(epoch, name, acc))

        name, val_acc = test(ctx, val_data, net)
        print('[Epoch %d] Validation: %s=%f'%(epoch, name, val_acc))

        trial.report(val_acc, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    net.save_parameters('mnist.params')

    return val_acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=6000)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
