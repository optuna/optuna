from typing import Any

import skorch
import torch
from torch import nn

import optuna
from optuna.integration import SkorchPruningCallback
from optuna.testing.integration import DeterministicPruner


class ClassifierModule(nn.Module):
    def __init__(self) -> None:
        super(ClassifierModule, self).__init__()
        self.dense0 = nn.Linear(4, 8)

    def forward(self, X: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.dense0(X)


def test_skorch_pruning_callback() -> None:

    X, y = torch.zeros(5, 4), torch.zeros(5, dtype=torch.long)

    def objective(trial: optuna.trial.Trial) -> float:

        net = skorch.NeuralNetClassifier(
            ClassifierModule,
            max_epochs=10,
            lr=0.02,
            callbacks=[SkorchPruningCallback(trial, "valid_acc")],
        )

        net.fit(X, y)
        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0
