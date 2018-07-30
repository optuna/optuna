import chainer
import chainer.links as L
import pytest
import typing  # NOQA

import pfnopt
from pfnopt import storages


parametrize_storage = pytest.mark.parametrize(
    'storage_init_func',
    [storages.InMemoryStorage, lambda: storages.RDBStorage('sqlite:///:memory:')]
)


@parametrize_storage
def test_chainer_pruning_extension(storage_init_func):
    # type: (typing.Callable[[], storages.BaseStorage]) -> None

    def objective(trial):
        # type: (pfnopt.trial.Trial) -> float

        n_data = 64
        batchsize = 16
        model = L.Classifier(chainer.Sequential([L.Linear(None, 2)]))
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        # Classify an integer into odd or even.
        dataset = chainer.dataset.TupleDataset(list(range(n_data)),
                                               [i % 2 for i in range(n_data)])
        train_iter = chainer.iterators.SerialIterator(dataset, batchsize)
        updater = chainer.training.StandardUpdater(train_iter, optimizer)
        trainer = chainer.training.Trainer(updater, (1, 'epoch'))
        trainer.extend(
            pfnopt.integration.chainer.ChainerPruningExtension(trial, 'main/loss',
                                                               (1, 'epoch')))
        log_report_extension = chainer.training.extensions.LogReport(log_name=None)
        trainer.extend(log_report_extension)
        with pytest.raises(pfnopt.pruners.TrialPruned):
            trainer.run(show_loop_exception_msg=False)
        return 1.0 - log_report_extension.log[-1]['main/accuracy']

    class AllPruner(pfnopt.pruners.BasePruner):

        def prune(self, storage, study_id, trial_id, step):
            # type: (storages.BaseStorage, int, int, int) -> bool

            return True

    study = pfnopt.create_study(storage_init_func(), pruner=AllPruner())
    study.run(objective, n_trials=1)
