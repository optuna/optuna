from typing import Any

import pytest

import optuna
from optuna._imports import try_import
from optuna.integration import FastAIV2PruningCallback
from optuna.testing.pruners import DeterministicPruner


with try_import():
    from fastai.data.core import DataLoader
    from fastai.data.core import DataLoaders
    from fastai.learner import Learner
    from fastai.metrics import accuracy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

pytestmark = pytest.mark.integration


def _generate_dummy_dataset() -> "torch.utils.data.DataLoader":
    data = torch.zeros(3, 20, dtype=torch.float32)
    target = torch.zeros(3, dtype=torch.int64)
    dataset = torch.utils.data.TensorDataset(data, target)
    return DataLoader(dataset, batch_size=1)


@pytest.fixture(scope="session")
def tmpdir(tmpdir_factory: Any) -> Any:
    return tmpdir_factory.mktemp("fastai_integration_test")


def test_fastai_pruning_callback(tmpdir: Any) -> None:
    train_loader = _generate_dummy_dataset()
    test_loader = _generate_dummy_dataset()

    data = DataLoaders(train_loader, test_loader, path=tmpdir)

    def objective(trial: optuna.trial.Trial) -> float:
        model = nn.Sequential(nn.Linear(20, 1), nn.Sigmoid())
        learn = Learner(
            data,
            model,
            loss_func=F.nll_loss,
            metrics=[accuracy],
        )
        learn.fit(1, cbs=FastAIV2PruningCallback(trial))

        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0
