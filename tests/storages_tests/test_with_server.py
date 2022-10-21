from concurrent.futures import ProcessPoolExecutor
import os
from typing import Sequence

import numpy as np
import pytest

import optuna
from optuna.storages import BaseStorage
from optuna.trial import TrialState


_STUDY_NAME = "_test_multiprocess"


def f(x: float, y: float) -> float:
    return (x - 3) ** 2 + y


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    trial.report(x, 0)
    trial.report(y, 1)
    trial.set_user_attr("x", x)
    trial.set_system_attr("y", y)
    return f(x, y)


def get_storage(storage_url: str, storage_mode: str) -> BaseStorage:
    if storage_mode == "RDB":
        if storage_url.startswith("redis"):
            return optuna.storages.RedisStorage(url=storage_url)
        else:
            return optuna.storages.RDBStorage(url=storage_url)
    elif storage_mode == "journal-redis":
        journal_redis_storage = optuna.storages.JournalRedisStorage(storage_url)
        return optuna.storages.JournalStorage(journal_redis_storage)
    else:
        assert False, f"The mode {storage_mode} is not supported."


def run_optimize(study_name: str, storage_url: str, storage_mode: str, n_trials: int) -> None:
    # Create a study
    study = optuna.load_study(
        study_name=study_name,
        storage=get_storage(storage_url, storage_mode),
    )
    # Run optimization
    study.optimize(objective, n_trials=n_trials)


@pytest.fixture
def storage_url() -> str:
    if "TEST_DB_URL" not in os.environ:
        pytest.skip("This test requires TEST_DB_URL.")
    storage_url = os.environ["TEST_DB_URL"]

    if "TEST_DB_MODE" not in os.environ:
        storage_mode = "RDB"
    else:
        storage_mode = os.environ["TEST_DB_MODE"]

    try:
        optuna.study.delete_study(
            study_name=_STUDY_NAME, storage=get_storage(storage_url, storage_mode)
        )
    except KeyError:
        pass
    return storage_url


@pytest.fixture
def storage_mode() -> str:
    if "TEST_DB_MODE" not in os.environ:
        storage_mode = "RDB"
    else:
        storage_mode = os.environ["TEST_DB_MODE"]
    return storage_mode


def _check_trials(trials: Sequence[optuna.trial.FrozenTrial]) -> None:
    # Check trial states.
    assert all(trial.state == TrialState.COMPLETE for trial in trials)

    # Check trial values and params.
    assert all("x" in trial.params for trial in trials)
    assert all("y" in trial.params for trial in trials)
    assert all(
        np.isclose(
            np.asarray([trial.value for trial in trials]),
            [f(trial.params["x"], trial.params["y"]) for trial in trials],
            atol=1e-4,
        ).tolist()
    )

    # Check intermediate values.
    assert all(len(trial.intermediate_values) == 2 for trial in trials)
    assert all(trial.params["x"] == trial.intermediate_values[0] for trial in trials)
    assert all(trial.params["y"] == trial.intermediate_values[1] for trial in trials)

    # Check attrs.
    assert all(
        np.isclose(
            [trial.user_attrs["x"] for trial in trials],
            [trial.params["x"] for trial in trials],
            atol=1e-4,
        ).tolist()
    )
    assert all(
        np.isclose(
            [trial.system_attrs["y"] for trial in trials],
            [trial.params["y"] for trial in trials],
            atol=1e-4,
        ).tolist()
    )


def test_loaded_trials(storage_url: str, storage_mode: str) -> None:
    # Please create the tables by placing this function before the multi-process tests.

    N_TRIALS = 20
    study = optuna.create_study(
        study_name=_STUDY_NAME, storage=get_storage(storage_url, storage_mode)
    )
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS)

    trials = study.trials
    assert len(trials) == N_TRIALS

    _check_trials(trials)

    # Create a new study to confirm the study can load trial properly.
    loaded_study = optuna.load_study(
        study_name=_STUDY_NAME, storage=get_storage(storage_url, storage_mode)
    )
    _check_trials(loaded_study.trials)


@pytest.mark.parametrize(
    "input_value,expected",
    [
        (float("inf"), float("inf")),
        (-float("inf"), -float("inf")),
    ],
)
def test_store_infinite_values(
    input_value: float, expected: float, storage_url: str, storage_mode: str
) -> None:

    storage = get_storage(storage_url, storage_mode)
    study_id = storage.create_new_study()
    trial_id = storage.create_new_trial(study_id)
    storage.set_trial_intermediate_value(trial_id, 1, input_value)
    storage.set_trial_state_values(trial_id, state=TrialState.COMPLETE, values=(input_value,))
    assert storage.get_trial(trial_id).value == expected
    assert storage.get_trial(trial_id).intermediate_values[1] == expected


def test_store_nan_intermediate_values(storage_url: str, storage_mode: str) -> None:

    storage = get_storage(storage_url, storage_mode)
    study_id = storage.create_new_study()
    trial_id = storage.create_new_trial(study_id)

    value = float("nan")
    storage.set_trial_intermediate_value(trial_id, 1, value)

    got_value = storage.get_trial(trial_id).intermediate_values[1]
    assert np.isnan(got_value)


def test_multiprocess(storage_url: str, storage_mode: str) -> None:
    n_workers = 8
    n_trials = 20
    study_name = _STUDY_NAME
    optuna.create_study(storage=get_storage(storage_url, storage_mode), study_name=study_name)
    with ProcessPoolExecutor(n_workers) as pool:
        pool.map(
            run_optimize, *zip(*[[study_name, storage_url, storage_mode, n_trials]] * n_workers)
        )

    study = optuna.load_study(
        study_name=study_name, storage=get_storage(storage_url, storage_mode)
    )

    trials = study.trials
    assert len(trials) == n_workers * n_trials

    _check_trials(trials)
