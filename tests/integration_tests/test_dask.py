import asyncio
from contextlib import contextmanager
import tempfile
import time
from typing import Iterator

from distributed import Client
from distributed import Scheduler
from distributed import SchedulerPlugin
from distributed import Worker
from distributed.utils_test import clean
from distributed.utils_test import gen_cluster
import numpy as np
import pytest

import optuna
from optuna.distributions import FloatDistribution
from optuna.integration.dask import _OptunaSchedulerExtension
from optuna.integration.dask import DaskStorage
from optuna.integration.dask import DaskStudy
from optuna.trial import Trial
from optuna.trial import TrialState


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
    x = trial.suggest_float("x", -10, 10)
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
        optuna.integration.dask.create_study()


def test_no_client_informative_error() -> None:
    with pytest.raises(ValueError, match="No global client found"):
        optuna.integration.dask.create_study()


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


def test_dask_create_study(client: Client) -> None:
    study = optuna.integration.dask.create_study()
    assert type(study) is DaskStudy
    assert type(study._storage) is DaskStorage


@pytest.mark.parametrize("storage_specifier", STORAGE_MODES)
def test_study_optimize(client: Client, storage_specifier: str) -> None:
    with get_storage_url(storage_specifier) as url:
        study = optuna.integration.dask.create_study(storage=url)
        study.optimize(objective, n_trials=10)
        assert len(study.trials) == 10


def test_study_optimize_on_cluster(client: Client) -> None:
    # This test is a sanity check that `DaskStudy.optimize` tasks
    # actually run on the Dask cluster.

    # Create a plugin which logs all the task keys run on the cluster
    # We'll later check that the expected number of optuna optimization
    # trial tasks get logged.
    class LogTaskKeys(SchedulerPlugin):
        def start(self, scheduler: Scheduler):  # type: ignore
            self.scheduler = scheduler

        def transition(self, key, start, finish, *args, **kwargs):  # type: ignore
            if start == "processing" and finish == "memory":
                self.scheduler.log_event("completed-task-keys", key)

    plugin = LogTaskKeys()
    client.register_scheduler_plugin(plugin)

    study = optuna.integration.dask.create_study()
    study.optimize(objective, n_trials=10)

    events = client.get_events("completed-task-keys")
    assert len(events) == 10
    assert all(event[1].startswith("optuna-optimize-trial-") for event in events)


def test_study_optimize_timeout(client: Client) -> None:
    study = optuna.integration.dask.create_study()
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

    study = optuna.integration.dask.create_study(direction=direction)
    study.optimize(objective, n_trials=10)

    # Ensure that study.best_value matches up with the expected value from
    # the trials DataFrame
    trials_value = study.trials_dataframe()["value"]
    if direction == "maximize":
        expected = trials_value.max()
    else:
        expected = trials_value.min()

    np.testing.assert_allclose(expected, study.best_value)


def test_study_set_and_get_user_attrs(client: Client) -> None:
    study = optuna.integration.dask.create_study()
    study.set_user_attr("dataset", "MNIST")
    assert study.user_attrs["dataset"] == "MNIST"


def test_study_set_and_get_system_attrs(client: Client) -> None:
    study = optuna.integration.dask.create_study()
    study.set_system_attr("system_message", "test")
    assert study.system_attrs["system_message"] == "test"


def test_study_ask_tell(client: Client) -> None:
    study = optuna.integration.dask.create_study()
    assert len(study.trials) == 0

    trial = study.ask()
    assert len(study.trials) == 1
    assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 0

    study.tell(trial, 1.0)
    assert len(study.trials) == 1
    assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 1


def test_study_trials_dataframe(client: Client) -> None:
    pd = pytest.importorskip("pandas")

    study = optuna.integration.dask.create_study()

    # Returns an empty DataFrame initially
    df = study.trials_dataframe()
    assert df.empty

    n_trials = 7
    study.optimize(objective, n_trials=n_trials)

    df = study.trials_dataframe()
    pd.testing.assert_series_equal(
        df.value, pd.Series(t.value for t in study.trials), check_names=False
    )
    pd.testing.assert_series_equal(
        df.params_x, pd.Series(t.params["x"] for t in study.trials), check_names=False
    )
    pd.testing.assert_series_equal(df.number, pd.Series(range(n_trials)), check_names=False)


def test_study_add_trial(client: Client) -> None:
    study = optuna.integration.dask.create_study()
    assert len(study.trials) == 0

    trial = optuna.trial.create_trial(
        params={"x": 2.0},
        distributions={"x": FloatDistribution(0, 10)},
        value=4.0,
    )

    study.add_trial(trial)
    assert len(study.trials) == 1


def test_study_add_trials(client: Client) -> None:
    study = optuna.integration.dask.create_study()
    study.optimize(objective, n_trials=3)
    assert len(study.trials) == 3

    other_study = optuna.integration.dask.create_study()
    other_study.add_trials(study.trials)
    assert len(other_study.trials) == len(study.trials)

    other_study.optimize(objective, n_trials=2)
    assert len(other_study.trials) == len(study.trials) + 2


def test_study_enqueue_trial(client: Client) -> None:
    study = optuna.integration.dask.create_study()
    study.enqueue_trial({"x": 5})
    study.enqueue_trial({"x": 0}, user_attrs={"memo": "optimal"})
    study.optimize(objective, n_trials=2)

    assert study.trials[0].params == {"x": 5}
    assert study.trials[1].params == {"x": 0}
    assert study.trials[1].user_attrs == {"memo": "optimal"}
