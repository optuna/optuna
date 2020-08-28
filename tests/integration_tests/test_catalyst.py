import shutil
import tempfile

import catalyst
import pytest
import torch

import optuna
from optuna.integration import CatalystPruningCallback
from optuna.testing.integration import DeterministicPruner


def test_catalyst_pruning_callback_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        CatalystPruningCallback(None)  # type: ignore


def test_catalyst_pruning_callback() -> None:
    data = torch.zeros(3, 4, dtype=torch.float32)
    target = torch.zeros(3, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(data, target)

    loaders = {
        "train": torch.utils.data.DataLoader(dataset, batch_size=1),
        "valid": torch.utils.data.DataLoader(dataset, batch_size=1),
    }

    def objective(trial: optuna.trial.Trial) -> float:
        model = torch.nn.Linear(4, 1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        dirpath = tempfile.mkdtemp()

        runner = catalyst.dl.SupervisedRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            logdir=dirpath,
            num_epochs=2,
            verbose=True,
            callbacks=[CatalystPruningCallback(trial, metric="loss")],
        )

        shutil.rmtree(dirpath)

        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0
