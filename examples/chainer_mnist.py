import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


N_TRAIN_EXAMPLES = 1000
N_TEST_EXAMPLES = 100
BATCHSIZE = 32
EPOCH = 10


class MLP(chainer.Chain):

    def __init__(self, client):
        super(MLP, self).__init__()

        n_units_l1 = int(client.sample_loguniform('n_units_l1', 16, 256))
        n_units_l2 = int(client.sample_loguniform('n_units_l2', 16, 256))
        n_units_l3 = int(client.sample_loguniform('n_units_l3', 16, 256))

        with self.init_scope():
            self.l1 = L.Linear(None, n_units_l1)
            self.l2 = L.Linear(None, n_units_l2)
            self.l3 = L.Linear(None, n_units_l3)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def obj(client):
    # Dataset
    rng = np.random.RandomState(0)
    train, test = chainer.datasets.get_mnist()
    train = chainer.datasets.SubDataset(
        train, 0, N_TRAIN_EXAMPLES, order=rng.permutation(len(train)))
    test = chainer.datasets.SubDataset(
        test, 0, N_TEST_EXAMPLES, order=rng.permutation(len(test)))
    train_iter = chainer.iterators.SerialIterator(train, BATCHSIZE)
    test_iter = chainer.iterators.SerialIterator(test, BATCHSIZE, repeat=False, shuffle=False)

    # Model
    model = L.Classifier(MLP(client))

    # Optimizer
    optimizer_name = client.sample_categorical('optimizer', ['Adam', 'MomentumSGD'])
    if optimizer_name == 'Adam':
        lr = client.sample_loguniform('adam_apha', 1e-5, 1e-1)
        optimizer = chainer.optimizers.Adam(alpha=lr)
    else:
        lr = client.sample_loguniform('lr', 1e-5, 1e-1)
        optimizer = chainer.optimizers.MomentumSGD(lr=lr)
    optimizer.setup(model)

    # Trainer
    updater = chainer.training.StandardUpdater(train_iter, optimizer)
    trainer = chainer.training.Trainer(updater, (EPOCH, 'epoch'))
    trainer.extend(chainer.training.extensions.Evaluator(test_iter, model))
    log_report_extension = chainer.training.extensions.LogReport(log_name=None)
    trainer.extend(chainer.training.extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(log_report_extension)

    # Run!
    trainer.run()
    return 1.0 - log_report_extension.log[-1]['validation/main/accuracy']  # Validation error


if __name__ == '__main__':
    import pfnopt
    pfnopt.minimize(obj, n_trials=100)
