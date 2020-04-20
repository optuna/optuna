import torch
from catalyst.dl import SupervisedRunner

import optuna
from optuna.integration import CatalystPruningCallback
from optuna.testing.integration import DeterministicPruner
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Dict  # NOQA
    from typing import List  # NOQA
    from typing import Union  # NOQA

def test_catalyst_pruning_callback():
    # type: () -> None

    data = torch.zeros(3, 4, dtype=torch.float32)
    target = torch.zeros(3, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(data, target)

    loaders = {
        "train": torch.utils.data.DataLoader(dataset, batch_size=1),
        "valid": torch.utils.data.DataLoader(dataset, batch_size=1),
    }

    def objective(trial):
        # type: (optuna.trial.Trial) -> None
        model = torch.nn.Linear(4, 1)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        runner = SupervisedRunner()
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            logdir='./logdir',
            num_epochs=2,
            verbose=True,
            callbacks=[
                        CatalystPruningCallback(trial, metric="loss"),
                      ],
        )

        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0
