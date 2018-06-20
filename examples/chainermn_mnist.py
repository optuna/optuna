"""
PFNOpt example that optimizes multi-layer perceptrons using ChainerMN.

In this example, we optimize the validation accuracy of hand-written digit recognition using
ChainerMN and MNIST, where architecture of neural network is optimized.

ChainerMN and it's PFNOpt integration are supposed to be invoked via MPI.
    $ mpirun -n 2 -- python chainermn_mnist.py

"""

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
import chainermn
import tempfile


BATCHSIZE = 128
EPOCH = 10


def create_model(trial):
    # We optimize the numbers of layers and their units.
    n_layers = int(trial.suggest_uniform('n_layers', 1, 4))

    layers = []
    for i in range(n_layers):
        n_units = int(trial.suggest_loguniform('n_units_l{}'.format(i), 4, 128))
        layers.append(L.Linear(None, n_units))
        layers.append(F.relu)
    layers.append(L.Linear(None, 10))

    return chainer.Sequential(*layers)


def objective(trial):
    # Sample an architecture.
    model = L.Classifier(create_model(trial))

    # Setup optimizer.
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model)
    optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)

    # Setup dataset and iterator.
    train, test = chainer.datasets.get_mnist()

    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    test = chainermn.scatter_dataset(test, comm)

    train_iter = chainer.iterators.SerialIterator(train, BATCHSIZE, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(test, BATCHSIZE, repeat=False, shuffle=False)

    # Setup trainer.
    updater = chainer.training.StandardUpdater(train_iter, optimizer)
    trainer = chainer.training.Trainer(updater, (EPOCH, 'epoch'))

    if comm.rank == 0:
        trainer.extend(chainer.training.extensions.ProgressBar())

    # Run training.
    trainer.run()

    # Evaluate.
    evaluator = chainer.training.extensions.Evaluator(test_iter, model)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    report = evaluator()

    return 1.0 - report['main/accuracy']


if __name__ == '__main__':
    import pfnopt

    comm = chainermn.create_communicator('naive')

    print('Number of nodes: ', comm.size)

    if comm.rank == 0:
        # ChainerMN integration supports only RDB backend.
        sqlite_file = tempfile.NamedTemporaryFile()
        sqlite_url = 'sqlite:///{}'.format(sqlite_file.name)

        study_uuid = pfnopt.create_study(storage=sqlite_url).study_uuid
    else:
        study_uuid, sqlite_url = None, None

    # Please make sure common study and storage are shared among nodes.
    study_uuid = comm.bcast_obj(study_uuid)
    sqlite_url = comm.bcast_obj(sqlite_url)

    # Run optimization!
    study = pfnopt.integration.minimize_chainermn(
        objective, study_uuid, comm, storage=sqlite_url, n_trials=25)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    if comm.rank == 0:
        sqlite_file.close()
