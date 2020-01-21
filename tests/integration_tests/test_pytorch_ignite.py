from ignite.engine import Engine
from mock import patch
import pytest

import optuna
from optuna.testing.integration import create_running_trial
from optuna.testing.integration import DeterministicPruner
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Iterable  # NOQA


def test_pytorch_ignite_pruning_handler():
    # type: () -> None

    def update(engine, batch):
        # type: (Engine, Iterable) -> None

        pass

    trainer = Engine(update)
    pruning_evaluator = Engine(update)

    # The pruner is activated.
    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = create_running_trial(study, 1.0)

    handler = optuna.integration.PyTorchIgnitePruningHandler(trial, 'accuracy', trainer)
    with patch.object(trainer, 'state', epoch=3, metrics={}):
        with patch.object(pruning_evaluator, 'state', epoch=1, metrics={'accuracy': 1}):
            with pytest.raises(optuna.exceptions.TrialPruned):
                handler(pruning_evaluator)
            assert study.trials[0].intermediate_values == {3: 1}

    # The pruner is not activated.
    study = optuna.create_study(pruner=DeterministicPruner(False))
    trial = create_running_trial(study, 1.0)

    handler = optuna.integration.PyTorchIgnitePruningHandler(trial, 'accuracy', trainer)
    with patch.object(trainer, 'state', epoch=5, metrics={}):
        with patch.object(pruning_evaluator, 'state', epoch=1, metrics={'accuracy': 2}):
            handler(pruning_evaluator)
            assert study.trials[0].intermediate_values == {5: 2}
