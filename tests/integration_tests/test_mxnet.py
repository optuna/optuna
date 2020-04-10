import mxnet as mx
import numpy as np
import pytest

import optuna
from optuna.integration.mxnet import MXNetPruningCallback
from optuna.testing.integration import DeterministicPruner
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Union  # NOQA


def test_mxnet_pruning_callback():
    # type: () -> None

    def objective(trial, eval_metric):
        # type: (optuna.trial.Trial, Union[list, str]) -> float

        # Symbol
        data = mx.symbol.Variable("data")
        data = mx.symbol.FullyConnected(data=data, num_hidden=1)
        data = mx.symbol.Activation(data=data, act_type="sigmoid")
        mlp = mx.symbol.SoftmaxOutput(data=data, name="softmax")

        # Optimizer
        optimizer = mx.optimizer.RMSProp()

        # Dataset
        train_data = mx.io.NDArrayIter(
            data=np.zeros((16, 20), np.float32),
            label=np.zeros((16,), np.int32),
            batch_size=1,
            shuffle=True,
        )

        eval_data = mx.io.NDArrayIter(
            data=np.zeros((5, 20), np.float32),
            label=np.zeros((5,), np.int32),
            batch_size=1,
            shuffle=True,
        )

        model = mx.mod.Module(symbol=mlp)
        model.fit(
            train_data=train_data,
            eval_data=eval_data,
            eval_metric=eval_metric,
            optimizer=optimizer,
            num_epoch=1,
            eval_end_callback=MXNetPruningCallback(trial, "accuracy"),
        )
        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(lambda trial: objective(trial, "accuracy"), n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(lambda trial: objective(trial, "accuracy"), n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].value == 1.0

    with pytest.raises(ValueError):
        objective(optuna.trial.Trial(study, 0), [])
        objective(optuna.trial.Trial(study, 0), ["mae"])

    study.optimize(lambda trial: objective(trial, ["accuracy", "mae"]), n_trials=1)
