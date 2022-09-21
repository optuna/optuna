import asyncio
from contextlib import contextmanager
import tempfile
import time
from typing import Iterator

from distributed import Client
from distributed import Scheduler
from distributed import Worker
from distributed.utils_test import clean
from distributed.utils_test import gen_cluster
import numpy as np
import pytest

import optuna
from optuna.integration.dask import _OptunaSchedulerExtension
from optuna.integration.dask import DaskStorage
from optuna.integration.dask import DaskStudy
from optuna.trial import Trial


# Ensure experimental warnings related to the Dask integration
# aren't included in the pytest warning summary for tests in this module
pytestmark = pytest.mark.filterwarnings(
    "ignore:DaskStorage is experimental",
    "ignore:DaskStudy is experimental",
)


STORAGE_MODES = ["inmemory", "sqlite"]


@contextmanager
def get_storage_url(specifier: str) -> Iterator:
    tmpfile = None
    try:
        if specifier == "inmemory":
            url = None
        elif specifier == "sqlite":
            tmpfile = tempfile.NamedTemporaryFile()
            url = "sqlite:///{}".format(tmpfile.name)
        else:
            raise ValueError(
                "Invalid specifier entered. Was expecting 'inmemory' or 'sqlite'"
                f"but got {specifier} instead"
            )
        yield url
    finally:
        if tmpfile is not None:
            tmpfile.close()


def objective(trial: Trial) -> float:
    x = trial.suggest_uniform("x", -10, 10)
    return (x - 2) ** 2


def objective_slow(trial: Trial) -> float:
    time.sleep(2)
    return objective(trial)


@pytest.fixture
def client() -> Client:
    with clean():
        with Client(dashboard_address=":0") as client:
            yield client


def test_experimental(client: Client) -> None:
    with pytest.warns(optuna._experimental.ExperimentalWarning):
        DaskStorage()


@gen_cluster(client=True)
async def test_daskstorage_registers_extension(
    c: Client, s: Scheduler, a: Worker, b: Worker
) -> None:
    assert "optuna" not in s.extensions
    await DaskStorage()
    assert "optuna" in s.extensions
    assert type(s.extensions["optuna"]) is _OptunaSchedulerExtension


@gen_cluster(client=True)
async def test_name(c: Client, s: Scheduler, a: Worker, b: Worker) -> None:
    await DaskStorage(name="foo")
    ext = s.extensions["optuna"]
    assert len(ext.storages) == 1
    assert type(ext.storages["foo"]) is optuna.storages.InMemoryStorage

    await DaskStorage(name="bar")
    assert len(ext.storages) == 2
    assert type(ext.storages["bar"]) is optuna.storages.InMemoryStorage


def test_name_unique(client: Client) -> None:
    s1 = DaskStorage()
    s2 = DaskStorage()
    assert s1.name != s2.name


def test_create_study_daskstudy(client: Client) -> None:
    storage = DaskStorage()
    study = optuna.create_study(storage=storage)
    study = DaskStudy(study)
    assert type(study) is DaskStudy
    assert type(study._storage) is DaskStorage


@pytest.mark.parametrize("storage_specifier", STORAGE_MODES)
def test_daskstudy_optimize(client: Client, storage_specifier: str) -> None:
    with get_storage_url(storage_specifier) as url:
        storage = DaskStorage(url)
        study = optuna.create_study(storage=storage)
        study = DaskStudy(study)
        study.optimize(objective, n_trials=10)
        assert len(study.trials) == 10


def test_daskstudy_optimize_timeout(client: Client) -> None:
    study = optuna.create_study(storage=DaskStorage())
    study = DaskStudy(study)
    with pytest.raises(asyncio.TimeoutError):
        study.optimize(objective_slow, n_trials=10, timeout=0.1)


@pytest.mark.parametrize("storage_specifier", STORAGE_MODES)
def test_get_base_storage(client: Client, storage_specifier: str) -> None:
    with get_storage_url(storage_specifier) as url:
        dask_storage = DaskStorage(url)
        storage = dask_storage.get_base_storage()
        expected_type = type(optuna.storages.get_storage(url))
        assert type(storage) is expected_type


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
def test_study_direction_best_value(client: Client, direction: str) -> None:
    # Regression test for https://github.com/jrbourbeau/dask-optuna/issues/15
    pytest.importorskip("pandas")
    dask_storage = DaskStorage()
    study = optuna.create_study(storage=dask_storage, direction=direction)
    study = DaskStudy(study)
    study.optimize(objective, n_trials=10)

    # Ensure that study.best_value matches up with the expected value from
    # the trials DataFrame
    trials_value = study.trials_dataframe()["value"]
    if direction == "maximize":
        expected = trials_value.max()
    else:
        expected = trials_value.min()

    np.testing.assert_allclose(expected, study.best_value)
