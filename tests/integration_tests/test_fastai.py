from functools import partial
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from fastai.basic_data import DataBunch
from fastai.basic_train import Learner
from fastai.metrics import accuracy
import numpy as np
import pytest

import optuna
from optuna.integration import FastaiPruningCallback
from optuna.testing.integration import create_running_trial
from optuna.testing.integration import DeterministicPruner


# https://docs.fast.ai/basic_data.html#Using-a-custom-Dataset-in-fastai
class ArrayDataset(Dataset):
    "Sample numpy array dataset"

    def __init__(self, x, y):
        self.x, self.y = x, y
        self.c = 2

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


@pytest.fixture(scope='session')
def tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('fastai_integration_test')


def test_fastai_pruning_callback(tmpdir):
    # type: () -> None

    train_x = np.zeros((16, 20), np.float32)
    train_y = np.zeros((16, ), np.int64)
    valid_x = np.zeros((4, 20), np.float32)
    valid_y = np.zeros((4, ), np.int64)
    train_ds = ArrayDataset(train_x, train_y)
    valid_ds = ArrayDataset(valid_x, valid_y)

    data_bunch = DataBunch.create(
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=None,
        path=tmpdir,
        bs=1  # batch size
    )

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        model = nn.Sequential(nn.Linear(20, 1), nn.Sigmoid())
        learn = Learner(data_bunch, model, metrics=[accuracy])

        learn.fit(13, callbacks=[FastaiPruningCallback(learn, trial, 'valid_loss')])

        return 1.0

    study = optuna.create_study(pruner=DeterministicPruner(True))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.structs.TrialState.PRUNED

    study = optuna.create_study(pruner=DeterministicPruner(False))
    study.optimize(objective, n_trials=1)
    assert study.trials[0].state == optuna.structs.TrialState.COMPLETE
    assert study.trials[0].value == 1.0


def test_fastai_pruning_callback_observation_isnan(tmpdir):
    # type: () -> None

    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = create_running_trial(study, 1.0)

    # FastaiPruningCallback requires `fastai.basic_train.Learner`.
    train_x = np.zeros((1, 20), np.float32)
    train_y = np.zeros((1, ), np.int64)
    valid_x = np.zeros((1, 20), np.float32)
    valid_y = np.zeros((1, ), np.int64)
    train_ds = ArrayDataset(train_x, train_y)
    valid_ds = ArrayDataset(valid_x, valid_y)

    data_bunch = DataBunch.create(
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=None,
        path=tmpdir,
        bs=1  # batch size
    )

    model = nn.Sequential(nn.Linear(20, 1), nn.Sigmoid())
    learn = Learner(data_bunch, model, metrics=[accuracy])

    callback = FastaiPruningCallback(learn)
    callback.register_trial_monitor(trial, 'valid_loss')

    # N.B. This is necessary to set `callback._index_to_monitor`.
    learn.fit(1)

    with pytest.raises(optuna.structs.TrialPruned):
        callback.on_epoch_end(1, torch.from_numpy(np.ones((1, ))), [1.0])

    with pytest.raises(optuna.structs.TrialPruned):
        callback.on_epoch_end(1, torch.from_numpy(np.ones((1, ))), [float('nan')])
