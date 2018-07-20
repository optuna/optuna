import chainer
import chainer.functions as F
import chainer.links as L
import numpy
import pytest
import typing  # NOQA

import pfnopt
from pfnopt import storages
from pfnopt.study import create_study


parametrize_storage = pytest.mark.parametrize(
    'storage_init_func',
    [storages.InMemoryStorage, lambda: storages.RDBStorage('sqlite:///:memory:')]
)


def create_model(trial):
    # type: (pfnopt.trial.Trial) -> float

    layers = []
    n_units = trial.suggest_int('n_units', 1, 10)
    layers.append(L.Linear(None, n_units))
    layers.append(F.relu)
    layers.append(L.Linear(None, 2))
    return chainer.Sequential(*layers)


class Dataset(chainer.dataset.DatasetMixin):

    def __init__(self, values):
        # type: (typing.List[int]) -> None

        self.values = values

    def __len__(self):
        # type: () -> int

        return len(self.values)

    def get_example(self, i):
        # type: (int) -> typing.Tuple[numpy.ndarray, int]

        return numpy.array([self.values[i]], numpy.float32), numpy.int32(i % 2)


@parametrize_storage
def test_chainer_pruning_extension(storage_init_func):
    # type: (typing.Callable[[], storages.BaseStorage]) -> None

    def objective(trial):
        # type: (pfnopt.trial.Trial) -> float

        n_data = 64
        batchsize = 16
        model = L.Classifier(create_model(trial))
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        dataset = Dataset([i for i in range(n_data)])
        train_iter = chainer.iterators.SerialIterator(dataset, batchsize)
        test_iter = chainer.iterators.SerialIterator(dataset, batchsize,
                                                     repeat=False, shuffle=False)
        updater = chainer.training.StandardUpdater(train_iter, optimizer)
        trainer = chainer.training.Trainer(updater, (10, 'epoch'))
        trainer.extend(chainer.training.extensions.Evaluator(test_iter, model))
        trainer.extend(
            pfnopt.integration.chainer.ChainerPruningExtension(trial, 'validation/main/loss',
                                                               (1, 'epoch')))
        log_report_extension = chainer.training.extensions.LogReport(log_name=None)
        trainer.extend(log_report_extension)
        with pytest.raises(pfnopt.pruners.TrialPruned):
            trainer.run(show_loop_exception_msg=False)
        return 1.0 - log_report_extension.log[-1]['validation/main/accuracy']

    class AllPruner(pfnopt.pruners.BasePruner):

        def prune(self, storage, study_id, trial_id, step):
            # type: (storages.BaseStorage, int, int, int) -> bool

            return True

    study = create_study(storage_init_func(), pruner=AllPruner())
    study.run(objective, n_trials=1)
