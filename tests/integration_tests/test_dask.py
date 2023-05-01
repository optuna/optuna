from contextlib import contextmanager
import time
from typing import Iterator

import numpy as np
import pytest

import optuna
from optuna._imports import try_import
from optuna.integration.dask import _OptunaSchedulerExtension
from optuna.integration.dask import DaskStorage
from optuna.testing.tempfile_pool import NamedTemporaryFilePool
from optuna.trial import Trial


with try_import() as _imports:
    from distributed import Client
    from distributed import Scheduler
    from distributed import wait
    from distributed import Worker
    from distributed.utils_test import clean
    from distributed.utils_test import gen_cluster

pytestmark = pytest.mark.integration


STORAGE_MODES = ["inmemory", "sqlite"]


@contextmanager
def get_storage_url(specifier: str) -> Iterator:
    tmpfile = None
    try:
        if specifier == "inmemory":
            url = None
        elif specifier == "sqlite":
            tmpfile = NamedTemporaryFilePool().tempfile()
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
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


def objective_slow(trial: Trial) -> float:
    time.sleep(2)
    return objective(trial)


@pytest.fixture
def client() -> "Client":  # type: ignore[misc]
    with clean():
        with Client(dashboard_address=":0") as client:  # type: ignore[no-untyped-call]
            yield client


def test_experimental(client: "Client") -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        DaskStorage()


def test_no_client_informative_error() -> None:
    with pytest.raises(ValueError, match="No global client found"):
        DaskStorage()


def test_name_unique(client: "Client") -> None:
    s1 = DaskStorage()
    s2 = DaskStorage()
    assert s1.name != s2.name


@pytest.mark.parametrize("storage_specifier", STORAGE_MODES)
def test_study_optimize(client: "Client", storage_specifier: str) -> None:
    with get_storage_url(storage_specifier) as url:
        storage = DaskStorage(storage=url)
        study = optuna.create_study(storage=storage)
        assert not study.trials
        futures = [
            client.submit(  # type: ignore[no-untyped-call]
                study.optimize, objective, n_trials=1, pure=False
            )
            for _ in range(10)
        ]
        wait(futures)  # type: ignore[no-untyped-call]
        assert len(study.trials) == 10


@pytest.mark.parametrize("storage_specifier", STORAGE_MODES)
def test_get_base_storage(client: "Client", storage_specifier: str) -> None:
    with get_storage_url(storage_specifier) as url:
        dask_storage = DaskStorage(url)
        storage = dask_storage.get_base_storage()
        expected_type = type(optuna.storages.get_storage(url))
        assert type(storage) is expected_type


@pytest.mark.parametrize("direction", ["maximize", "minimize"])
def test_study_direction_best_value(client: "Client", direction: str) -> None:
    # Regression test for https://github.com/jrbourbeau/dask-optuna/issues/15
    pytest.importorskip("pandas")
    storage = DaskStorage()
    study = optuna.create_study(storage=storage, direction=direction)
    f = client.submit(study.optimize, objective, n_trials=10)  # type: ignore[no-untyped-call]
    wait(f)  # type: ignore[no-untyped-call]

    # Ensure that study.best_value matches up with the expected value from
    # the trials DataFrame
    trials_value = study.trials_dataframe()["value"]
    if direction == "maximize":
        expected = trials_value.max()
    else:
        expected = trials_value.min()

    np.testing.assert_allclose(expected, study.best_value)


if _imports.is_successful():

    @gen_cluster(client=True)
    async def test_daskstorage_registers_extension(
        c: "Client", s: "Scheduler", a: "Worker", b: "Worker"
    ) -> None:
        assert "optuna" not in s.extensions
        await DaskStorage()
        assert "optuna" in s.extensions
        assert type(s.extensions["optuna"]) is _OptunaSchedulerExtension

    @gen_cluster(client=True)
    async def test_name(c: "Client", s: "Scheduler", a: "Worker", b: "Worker") -> None:
        await DaskStorage(name="foo")
        ext = s.extensions["optuna"]
        assert len(ext.storages) == 1
        assert type(ext.storages["foo"]) is optuna.storages.InMemoryStorage

        await DaskStorage(name="bar")
        assert len(ext.storages) == 2
        assert type(ext.storages["bar"]) is optuna.storages.InMemoryStorage
