"""
Optuna example that optimizes multi-layer perceptrons using ChainerMN.

In this example, we optimize the validation accuracy of hand-written digit recognition using
ChainerMN and MNIST, where architecture of neural network is optimized.

ChainerMN and it's Optuna integration are supposed to be invoked via MPI. You can run this example
as follows:
    $ STORAGE_URL=sqlite:///example.db
    $ STUDY_NAME=`optuna create-study --direction maximize --storage $STORAGE_URL`
    $ mpirun -n 2 -- python chainermn_simple.py $STUDY_NAME $STORAGE_URL

"""
import sys

import chainer
import chainer.functions as F
import chainer.links as L
import chainermn
import numpy as np

import optuna


N_TRAIN_EXAMPLES = 3000
N_VALID_EXAMPLES = 1000
BATCHSIZE = 128
EPOCH = 10


def create_model(trial):
    # We optimize the numbers of layers and their units.
    n_layers = trial.suggest_int("n_layers", 1, 3)

    layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True)
        layers.append(L.Linear(None, n_units))
        layers.append(F.relu)
    layers.append(L.Linear(None, 10))

    return chainer.Sequential(*layers)


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial, comm):
    # Sample an architecture.
    model = L.Classifier(create_model(trial))

    # Setup optimizer.
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model)
    optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)

    # Setup dataset and iterator. Only worker 0 loads the whole dataset.
    # The dataset of worker 0 is evenly split and distributed to all workers.
    if comm.rank == 0:
        train, valid = chainer.datasets.get_mnist()
        rng = np.random.RandomState(0)
        train = chainer.datasets.SubDataset(
            train, 0, N_TRAIN_EXAMPLES, order=rng.permutation(len(train))
        )
        valid = chainer.datasets.SubDataset(
            valid, 0, N_VALID_EXAMPLES, order=rng.permutation(len(valid))
        )
    else:
        train, valid = None, None

    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    valid = chainermn.scatter_dataset(valid, comm)

    train_iter = chainer.iterators.SerialIterator(train, BATCHSIZE, shuffle=True)
    valid_iter = chainer.iterators.SerialIterator(valid, BATCHSIZE, repeat=False, shuffle=False)

    # Setup trainer.
    updater = chainer.training.StandardUpdater(train_iter, optimizer)
    trainer = chainer.training.Trainer(updater, (EPOCH, "epoch"))

    if comm.rank == 0:
        trainer.extend(chainer.training.extensions.ProgressBar())

    # Run training.
    trainer.run()

    # Evaluate.
    evaluator = chainer.training.extensions.Evaluator(valid_iter, model)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    report = evaluator()

    return report["main/accuracy"]


if __name__ == "__main__":
    # Please make sure common study and storage are shared among nodes.
    study_name = sys.argv[1]
    storage_url = sys.argv[2]

    study = optuna.load_study(study_name, storage_url)
    comm = chainermn.create_communicator("naive")
    if comm.rank == 0:
        print("Study name:", study_name)
        print("Storage URL:", storage_url)
        print("Number of nodes:", comm.size)

    # Run optimization!
    chainermn_study = optuna.integration.ChainerMNStudy(study, comm)
    chainermn_study.optimize(objective, n_trials=25)

    if comm.rank == 0:
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
