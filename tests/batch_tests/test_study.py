import itertools
import threading
import time
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
import uuid

import numpy as np
import pytest

import optuna
from optuna.batch.trial import BatchTrial
from optuna.testing.storage import StorageSupplier

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


def check_batch_study(study: optuna.batch.study.BatchStudy) -> None:
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


@pytest.mark.parametrize(
    "direction_str,direction",
    [
        ("minimize", optuna.study.StudyDirection.MINIMIZE),
        ("maximize", optuna.study.StudyDirection.MAXIMIZE),
    ],
)
def test_create_study(direction_str: str, direction: optuna.study.StudyDirection) -> None:
    batch_size = 4
    study = optuna.batch.create_study(direction=direction_str, batch_size=batch_size)
    assert study.batch_size == batch_size
    assert study.direction == direction


def test_load_study() -> None:
    batch_size = 4

    with StorageSupplier("sqlite") as storage:
        study_name = str(uuid.uuid4())

        with pytest.raises(KeyError):
            # Test loading an unexisting study.
            optuna.batch.study.load_study(
                study_name=study_name, storage=storage, batch_size=batch_size
            )

        # Create a new study.
        created_study = optuna.batch.study.create_study(
            study_name=study_name, storage=storage, batch_size=4
        )

        # Test loading an existing study.
        loaded_study = optuna.batch.study.load_study(
            study_name=study_name, storage=storage, batch_size=batch_size
        )
        assert created_study._study_id == loaded_study._study_id


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_optimize_trivial(batch_size: int) -> None:
    study = optuna.batch.create_study(batch_size=batch_size)
    study.optimize(func, n_batches=2)
    check_batch_study(study)
    assert len(study.trials) == batch_size * 2


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_batch_optimize_trivial_resume(batch_size: int) -> None:
    study = optuna.batch.create_study(batch_size=batch_size)
    study.optimize(func, n_batches=2)
    study.optimize(func, n_batches=2)
    check_batch_study(study)
    assert len(study.trials) == batch_size * 4


@pytest.mark.parametrize(
    "n_batches, n_jobs, storage_mode",
    itertools.product(
        (0, 1, 20),
        (1,),
        ["inmemory"],
    ),  # n_batches  # n_jobs  # storage_mode
)
def test_optimize_parallel(n_batches: int, n_jobs: int, storage_mode: str) -> None:
    f = Func()

    batch_size = 4

    with StorageSupplier(storage_mode) as storage:
        study = optuna.batch.create_study(storage=storage, batch_size=batch_size)
        study.optimize(f, n_batches=n_batches, n_jobs=n_jobs)
        assert f.n_calls == n_batches
        check_batch_study(study)


def test_study_user_attrs() -> None:
    study = optuna.batch.create_study(batch_size=4)

    study.set_user_attr("foo", "bar")
    assert study.user_attrs == {"foo": "bar"}

    study.set_user_attr("baz", "qux")
    assert study.user_attrs == {"foo": "bar", "baz": "qux"}

    study.set_user_attr("foo", "quux")
    assert study.user_attrs == {"foo": "quux", "baz": "qux"}


def test_study_system_attrs() -> None:
    study = optuna.batch.create_study(batch_size=4)

    study.set_system_attr("foo", "bar")
    assert study.system_attrs == {"foo": "bar"}

    study.set_system_attr("baz", "qux")
    assert study.system_attrs == {"foo": "bar", "baz": "qux"}

    study.set_system_attr("foo", "quux")
    assert study.system_attrs == {"foo": "quux", "baz": "qux"}


def test_enqueue_trial() -> None:
    batch_size = 4
    study = optuna.batch.create_study(batch_size=batch_size)
    for i in range(batch_size):
        study.enqueue_trial({"x": i})

    def objective(trial: optuna.batch.trial.BatchTrial) -> np.ndarray:
        x = trial.suggest_uniform("x", 0, 100)
        assert all(x == np.array(range(batch_size)))
        return np.zeros(batch_size)

    study.optimize(objective, n_batches=1)


def test_callbacks() -> None:
    batch_size = 2
    study = optuna.batch.create_study(batch_size=batch_size)

    def objective(trial: optuna.batch.trial.BatchTrial) -> np.ndarray:
        return trial.suggest_float("x", 0, 10)

    list0 = []
    list1 = []
    callbacks = [
        lambda _, trial: list0.append(trial.number),
        lambda _, trial: list1.append(trial.number),
    ]
    study.optimize(objective, n_batches=1, callbacks=callbacks)

    assert list0 == [0, 1]
    assert list1 == [0, 1]
