import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.metrics import Acc
import numpy as np
import pytest

import optuna
from optuna.integration import FastaiPruningCallback
from optuna.testing.integration import create_running_trial
from optuna.testing.integration import DeterministicPruner


@pytest.fixture(scope='session')
def test_fastai_pruning_callback(tmpdir_factory):
    # type: () -> None

    train_x = torch.from_numpy(np.zeros((16, 20), np.float32))
    train_y = torch.from_numpy(np.zeros((16, ), np.int32))
    valid_x = torch.from_numpy(np.zeros((4, 20), np.float32))
    valid_y = torch.from_numpy(np.zeros((4, ), np.int32))
    train_ds = TensorDataset(train_x, train_y)
    valid_ds = TensorDataset(valid_x, valid_y)

    data_bunch = DataBunch.create(
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=None,
        path=tmpdir_factory.mktemp('fastai_integration_test'),
        bs=1  # batch size
    )

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        model = nn.Sequential(nn.Linear(20, 1), F.sigmoid)
        learn = Learner(data_bunch, model, F.nll_loss, metrics=[Acc])

        learn.fit(1, 1e-3)

        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.structs.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.structs.TrialState.COMPLETE
    assert study.trials[0].value == 1.0


@pytest.fixture(scope='session')
def test_fastai_pruning_callback_observation_isnan(tmpdir_factory):
    # type: () -> None

    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = create_running_trial(study, 1.0)

    # FastaiPruningCallback requires `learn`, a bit similar to Keras Model.
    train_x = torch.from_numpy(np.zeros((1, 20), np.float32))
    train_y = torch.from_numpy(np.zeros((1, ), np.int32))
    valid_x = torch.from_numpy(np.zeros((1, 20), np.float32))
    valid_y = torch.from_numpy(np.zeros((1, ), np.int32))
    train_ds = TensorDataset(train_x, train_y)
    valid_ds = TensorDataset(valid_x, valid_y)

    data_bunch = DataBunch.create(
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=None,
        path=tmpdir_factory.mktemp('fastai_integration_test'),
        bs=1  # batch size
    )
    model = nn.Sequential(nn.Linear(20, 1), F.sigmoid)
    learn = Learner(data_bunch, model, F.nll_loss, metrics=[Acc])

    callback = FastaiPruningCallback(learn, trial, 'valid_loss')

    with pytest.raises(optuna.structs.TrialPruned):
        callback.on_epoch_end(1, torch.Tensor(1.0), [1.0])

    with pytest.raises(optuna.structs.TrialPruned):
        callback.on_epoch_end(1, torch.Tensor(1.0), [float('nan')])
