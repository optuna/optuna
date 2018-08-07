import chainer
import chainer.links as L
from chainer.training import triggers
import math
import numpy as np
import pytest
import typing  # NOQA

import pfnopt
from pfnopt.integration.chainer import ChainerPruningExtension


class FixedValueDataset(chainer.dataset.DatasetMixin):

    size = 16

    def __len__(self):
        # type: () -> int

        return self.size

    def get_example(self, i):
        # type: (int) -> typing.Tuple[np.ndarray, int]

        return np.array([1.0], np.float32), np.int32(0)


class FixedValuePruner(pfnopt.pruners.BasePruner):

    def __init__(self, is_pruning):
        # type: (bool) -> None

        self.is_pruning = is_pruning

    def prune(self, storage, study_id, trial_id, step):
        # type: (pfnopt.storages.BaseStorage, int, int, int) -> bool

        return self.is_pruning


def test_chainer_pruning_extension_trigger():
    study = pfnopt.create_study()
    trial = study._run_trial(func=lambda _: 1.0, catch=(Exception,))

    extension = ChainerPruningExtension(trial, 'main/loss', (1, 'epoch'))
    assert isinstance(extension.pruner_trigger, triggers.IntervalTrigger)
    extension = ChainerPruningExtension(trial, 'main/loss',
                                        triggers.IntervalTrigger(1, 'epoch'))
    assert isinstance(extension.pruner_trigger, triggers.IntervalTrigger)
    extension = ChainerPruningExtension(trial, 'main/loss',
                                        triggers.ManualScheduleTrigger(1, 'epoch'))
    assert isinstance(extension.pruner_trigger, triggers.ManualScheduleTrigger)

    with pytest.raises(TypeError):
        ChainerPruningExtension(trial, 'main/loss', triggers.TimeTrigger(1.))


def test_chainer_pruning_extension():
    # type: () -> None

    def objective(trial):
        # type: (pfnopt.trial.Trial) -> float

        model = L.Classifier(chainer.Sequential(L.Linear(None, 2)))
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        train_iter = chainer.iterators.SerialIterator(FixedValueDataset(), 16)
        updater = chainer.training.StandardUpdater(train_iter, optimizer)
        trainer = chainer.training.Trainer(updater, (1, 'epoch'))
        trainer.extend(
            pfnopt.integration.chainer.ChainerPruningExtension(trial, 'main/loss',
                                                               (1, 'epoch')))

        trainer.run(show_loop_exception_msg=False)
        return 1.0

    study = pfnopt.create_study(pruner=FixedValuePruner(True))
    study.run(objective, n_trials=1)
    assert study.trials[0].state == pfnopt.structs.TrialState.PRUNED

    study = pfnopt.create_study(pruner=FixedValuePruner(False))
    study.run(objective, n_trials=1)
    assert study.trials[0].state == pfnopt.structs.TrialState.COMPLETE
    assert study.trials[0].value == 1.0


def test_get_float_value():
    # type: () -> None

    study = pfnopt.create_study()
    trial = study._run_trial(func=lambda _: 1.0, catch=(Exception,))
    extension = pfnopt.integration.chainer.ChainerPruningExtension(trial, 'value', (1, 'epoch'))

    assert 1.0 == extension._get_float_value(1.0)
    assert 1.0 == extension._get_float_value(chainer.Variable(np.array([1.0])))
    assert math.isnan(extension._get_float_value(float('nan')))
    with pytest.raises(TypeError):
        extension._get_float_value([])
