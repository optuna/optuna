from collections import namedtuple
import math
import typing
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

import optuna
from optuna._imports import try_import
from optuna.integration.chainer import ChainerPruningExtension
from optuna.testing.pruners import DeterministicPruner


with try_import() as _imports:
    import chainer
    from chainer.dataset import DatasetMixin  # type: ignore[attr-defined]
    import chainer.links as L
    from chainer.training import triggers

if not _imports.is_successful():
    DatasetMixin = object  # NOQA

pytestmark = pytest.mark.integration


class FixedValueDataset(DatasetMixin):

    size = 16

    def __len__(self) -> int:

        return self.size

    def get_example(self, i: int) -> typing.Tuple[np.ndarray, np.signedinteger]:

        return np.array([1.0], np.float32), np.intc(0)


def test_chainer_pruning_extension_trigger() -> None:

    study = optuna.create_study()
    trial = study.ask()

    extension = ChainerPruningExtension(trial, "main/loss", (1, "epoch"))
    assert isinstance(
        extension._pruner_trigger, triggers.IntervalTrigger  # type: ignore[attr-defined]
    )
    extension = ChainerPruningExtension(
        trial, "main/loss", triggers.IntervalTrigger(1, "epoch")  # type: ignore[attr-defined]
    )
    assert isinstance(
        extension._pruner_trigger, triggers.IntervalTrigger  # type: ignore[attr-defined]
    )
    extension = ChainerPruningExtension(
        trial,
        "main/loss",
        triggers.ManualScheduleTrigger(1, "epoch"),  # type: ignore[attr-defined]
    )
    assert isinstance(
        extension._pruner_trigger, triggers.ManualScheduleTrigger  # type: ignore[attr-defined]
    )

    with pytest.raises(TypeError):
        ChainerPruningExtension(
            trial, "main/loss", triggers.TimeTrigger(1.0)  # type: ignore[attr-defined]
        )


def test_chainer_pruning_extension() -> None:
    @typing.no_type_check
    def objective(trial: optuna.trial.Trial) -> float:

        model = L.Classifier(chainer.Sequential(L.Linear(None, 2)))
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)

        train_iter = chainer.iterators.SerialIterator(FixedValueDataset(), 16)
        updater = chainer.training.StandardUpdater(train_iter, optimizer)
        trainer = chainer.training.Trainer(updater, (1, "epoch"))
        trainer.extend(
            optuna.integration.chainer.ChainerPruningExtension(trial, "main/loss", (1, "epoch"))
        )

        trainer.run(show_loop_exception_msg=False)
        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0


def test_chainer_pruning_extension_observation_nan() -> None:

    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = study.ask()
    extension = ChainerPruningExtension(trial, "main/loss", (1, "epoch"))

    MockTrainer = namedtuple("MockTrainer", ("observation", "updater"))
    MockUpdater = namedtuple("MockUpdater", ("epoch"))
    trainer = MockTrainer(observation={"main/loss": float("nan")}, updater=MockUpdater(1))

    with patch.object(extension, "_observation_exists", Mock(return_value=True)) as mock:
        with pytest.raises(optuna.TrialPruned):
            extension(trainer)
        assert mock.call_count == 1


def test_observation_exists() -> None:

    study = optuna.create_study()
    trial = study.ask()
    MockTrainer = namedtuple("MockTrainer", ("observation",))
    trainer = MockTrainer(observation={"OK": 0})

    # Trigger is deactivated. Return False whether trainer has observation or not.
    with patch.object(
        triggers.IntervalTrigger,  # type: ignore[attr-defined]
        "__call__",
        Mock(return_value=False),
    ) as mock:
        extension = ChainerPruningExtension(trial, "NG", (1, "epoch"))
        assert extension._observation_exists(trainer) is False
        extension = ChainerPruningExtension(trial, "OK", (1, "epoch"))
        assert extension._observation_exists(trainer) is False
        assert mock.call_count == 2

    # Trigger is activated. Return True if trainer has observation.
    with patch.object(
        triggers.IntervalTrigger, "__call__", Mock(return_value=True)  # type: ignore[attr-defined]
    ) as mock:
        extension = ChainerPruningExtension(trial, "NG", (1, "epoch"))
        assert extension._observation_exists(trainer) is False
        extension = ChainerPruningExtension(trial, "OK", (1, "epoch"))
        assert extension._observation_exists(trainer) is True
        assert mock.call_count == 2


def test_get_float_value() -> None:

    assert 1.0 == ChainerPruningExtension._get_float_value(1.0)
    assert 1.0 == ChainerPruningExtension._get_float_value(
        chainer.Variable(np.array([1.0]))  # type: ignore[attr-defined]
    )
    assert math.isnan(ChainerPruningExtension._get_float_value(float("nan")))
