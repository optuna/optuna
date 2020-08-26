import itertools
import threading
import time
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
import pytest

import optuna
from optuna import BatchStudy
from optuna.testing.storage import StorageSupplier
from optuna.trial import BatchTrial

STORAGE_MODES = [
    "inmemory",
    "sqlite",
    "cache",
    "redis",
]


def func(trial: BatchTrial, x_max: float = 1.0) -> np.ndarray:
    x = trial.suggest_float("x", -x_max, x_max)
    y = trial.suggest_float("y", 20, 30, log=True)
    z = trial.suggest_categorical("z", (-1.0, 1.0))
    assert isinstance(z, np.ndarray)
    return (x - 2) ** 2 + (y - 25) ** 2 + z


class Func(object):
    def __init__(self, sleep_sec: Optional[float] = None) -> None:

        self.n_calls = 0
        self.sleep_sec = sleep_sec
        self.lock = threading.Lock()
        self.x_max = 10.0

    def __call__(self, trial: BatchTrial) -> np.ndarray:

        with self.lock:
            self.n_calls += 1
            x_max = self.x_max
            self.x_max *= 0.9

        # Sleep for testing parallelism
        if self.sleep_sec is not None:
            time.sleep(self.sleep_sec)

        value = func(trial, x_max)
        check_batch_params(trial.params)
        return value


def check_batch_params(batch_params: Sequence[Dict[str, Any]]) -> None:
    for params in batch_params:
        check_params(params)


def check_params(params: Dict[str, Any]) -> None:
    assert sorted(params.keys()) == ["x", "y", "z"]


def check_value(value: Optional[float]) -> None:
    assert isinstance(value, float)
    assert -1.0 <= value <= 12.0 ** 2 + 5.0 ** 2 + 1.0


def check_frozen_trial(frozen_trial: optuna.trial.FrozenTrial) -> None:

    if frozen_trial.state == optuna.trial.TrialState.COMPLETE:
        check_params(frozen_trial.params)
        check_value(frozen_trial.value)


def check_batch_study(study: BatchStudy) -> None:
    for trial in study.trials:
        check_frozen_trial(trial)

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete_trials) == 0:
        with pytest.raises(ValueError):
            study.best_params
        with pytest.raises(ValueError):
            study.best_value
        with pytest.raises(ValueError):
            study.best_trial
    else:
        check_params(study.best_params)
        check_value(study.best_value)
        check_frozen_trial(study.best_trial)


def test_optimize() -> None:
    study = optuna.create_study()
    bstudy = BatchStudy(study, 4)

    def objective(trial: optuna.trial.Trial) -> float:
        return 1

    with pytest.raises(NotImplementedError):
        bstudy.optimize(objective)


def test_batch_optimize_trivial() -> None:
    study = optuna.create_study()
    batch_study = BatchStudy(study, batch_size=4)
    batch_study.batch_optimize(func, n_batches=2)
    check_batch_study(batch_study)


def test_batch_optimize_trivial_resume() -> None:
    batch_size = 6
    study = optuna.create_study()
    batch_study = BatchStudy(study, batch_size=batch_size)
    batch_study.batch_optimize(func, n_batches=2)
    batch_study.batch_optimize(func, n_batches=2)
    check_batch_study(batch_study)
    assert len(study.trials) == batch_size * 4


@pytest.mark.parametrize(
    "n_batches, n_jobs, storage_mode",
    itertools.product((0, 1, 20), (1,), ["inmemory"],),  # n_batches  # n_jobs  # storage_mode
)
def test_optimize_parallel(n_batches: int, n_jobs: int, storage_mode: str) -> None:
    f = Func()

    batch_size = 4

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        batch_study = BatchStudy(study, batch_size == batch_size)
        batch_study.batch_optimize(f, n_batches=n_batches, n_jobs=n_jobs)
        assert f.n_calls == n_batches
        check_batch_study(batch_study)
