import mxnet as mx
import numpy as np

import optuna
from optuna.integration.mxnet import MxnetPruningCallback
from optuna.testing.integration import DeterministicPruner


def test_mxnet_pruning_callback():
    # type: () -> None

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        # Symbol
        data = mx.symbol.Variable('data')
        data = mx.symbol.FullyConnected(data=data, num_hidden=1)
        data = mx.symbol.Activation(data=data, act_type="sigmoid")
        mlp = mx.symbol.SoftmaxOutput(data=data, name="softmax")

        # Optimizer
        optimizer = mx.optimizer.RMSProp()

        # Dataset
        train_data = mx.io.NDArrayIter(data=np.zeros((16, 20), np.float32),
                                       label=np.zeros((16,), np.int32),
                                       batch_size=1,
                                       shuffle=True)

        eval_data = mx.io.NDArrayIter(data=np.zeros((5, 20), np.float32),
                                      label=np.zeros((5,), np.int32),
                                      batch_size=1,
                                      shuffle=True)

        model = mx.mod.Module(symbol=mlp)
        model.fit(train_data=train_data,
                  eval_data=eval_data,
                  optimizer=optimizer,
                  num_epoch=1,
                  eval_end_callback=MxnetPruningCallback(trial, 'accuracy'))
        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.structs.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.structs.TrialState.COMPLETE
    assert study.trials[0].value == 1.0
