import chainer
import chainer.links as L
import pytest

import pfnopt


parametrize_observation = pytest.mark.parametrize(
    'observation_key',
    ['main/loss', 'validation/main/loss']
)


class Pruner(pfnopt.pruners.BasePruner):

    def __init__(self, is_pruning):
        # type: (bool) -> None

        self.is_pruning = is_pruning

    def prune(self, storage, study_id, trial_id, step):
        # type: (pfnopt.storages.BaseStorage, int, int, int) -> bool

        return self.is_pruning


@parametrize_observation
def test_chainer_pruning_extension(observation_key):
    # type: (str) -> None

    def objective(trial):
        # type: (pfnopt.trial.Trial) -> float

        model = L.Classifier(chainer.Sequential(L.Linear(None, 10)))
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        train, test = chainer.datasets.get_mnist()
        train = chainer.datasets.SubDataset(train, 0, 16)
        test = chainer.datasets.SubDataset(test, 0, 16)
        train_iter = chainer.iterators.SerialIterator(train, 16)
        test_iter = chainer.iterators.SerialIterator(test, 16, repeat=False, shuffle=False)
        updater = chainer.training.StandardUpdater(train_iter, optimizer)
        trainer = chainer.training.Trainer(updater, (1, 'epoch'))
        trainer.extend(chainer.training.extensions.Evaluator(test_iter, model))
        # Type of trainer.observation['main/loss'] is chainer.Variable
        # while type of trainer.observation['validation/main/loss'] is float.
        trainer.extend(
            pfnopt.integration.chainer.ChainerPruningExtension(trial, observation_key,
                                                               (1, 'epoch')))
        trainer.run(show_loop_exception_msg=False)
        return 1.0

    study = pfnopt.create_study(pruner=Pruner(True))
    study.run(objective, n_trials=1)
    assert study.trials[0].state == pfnopt.structs.TrialState.PRUNED

    study = pfnopt.create_study(pruner=Pruner(False))
    study.run(objective, n_trials=1)
    assert study.trials[0].state == pfnopt.structs.TrialState.COMPLETE
    assert study.trials[0].value == 1.0
