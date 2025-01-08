from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import os
import pickle

import numpy as np
import pytest

import optuna
from optuna.storages import BaseStorage
from optuna.storages.journal import JournalRedisBackend
from optuna.study import StudyDirection
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
    return f(x, y)


def get_storage() -> BaseStorage:
    if "TEST_DB_URL" not in os.environ:
        pytest.skip("This test requires TEST_DB_URL.")
    storage_url = os.environ["TEST_DB_URL"]
    storage_mode = os.environ.get("TEST_DB_MODE", "")

    storage: BaseStorage
    if storage_mode == "":
        storage = optuna.storages.RDBStorage(url=storage_url)
    elif storage_mode == "journal-redis":
        journal_redis_storage = JournalRedisBackend(storage_url)
        storage = optuna.storages.JournalStorage(journal_redis_storage)
    else:
        assert False, f"The mode {storage_mode} is not supported."

    return storage


def run_optimize(study_name: str, n_trials: int) -> None:
    storage = get_storage()
    # Create a study
    study = optuna.load_study(study_name=study_name, storage=storage)
    # Run optimization
    study.optimize(objective, n_trials=n_trials)


def _check_trials(trials: Sequence[optuna.trial.FrozenTrial]) -> None:
    # Check trial states.
    assert all(trial.state == TrialState.COMPLETE for trial in trials)

    # Check trial values and params.
    assert all("x" in trial.params for trial in trials)
    assert all("y" in trial.params for trial in trials)
    assert all(
        [
            condition
            for condition in np.isclose(
                np.asarray([trial.value for trial in trials]),
                [f(trial.params["x"], trial.params["y"]) for trial in trials],
                atol=1e-4,
            )
        ]
    )

    # Check intermediate values.
    assert all(len(trial.intermediate_values) == 2 for trial in trials)
    assert all(trial.params["x"] == trial.intermediate_values[0] for trial in trials)
    assert all(trial.params["y"] == trial.intermediate_values[1] for trial in trials)

    # Check attrs.
    assert all(
        [
            condition
            for condition in np.isclose(
                [trial.user_attrs["x"] for trial in trials],
                [trial.params["x"] for trial in trials],
                atol=1e-4,
            )
        ]
    )


def test_loaded_trials() -> None:
    # Please create the tables by placing this function before the multi-process tests.

    storage = get_storage()
    try:
        optuna.delete_study(study_name=_STUDY_NAME, storage=storage)
    except KeyError:
        pass

    N_TRIALS = 20
    study = optuna.create_study(study_name=_STUDY_NAME, storage=storage)
    # Run optimization
    study.optimize(objective, n_trials=N_TRIALS)

    trials = study.trials
    assert len(trials) == N_TRIALS

    _check_trials(trials)

    # Create a new study to confirm the study can load trial properly.
    loaded_study = optuna.load_study(study_name=_STUDY_NAME, storage=storage)
    _check_trials(loaded_study.trials)


@pytest.mark.parametrize(
    "input_value,expected",
    [
        (float("inf"), float("inf")),
        (-float("inf"), -float("inf")),
    ],
)
def test_store_infinite_values(input_value: float, expected: float) -> None:
    storage = get_storage()
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    trial_id = storage.create_new_trial(study_id)
    storage.set_trial_intermediate_value(trial_id, 1, input_value)
    storage.set_trial_state_values(trial_id, state=TrialState.COMPLETE, values=(input_value,))
    assert storage.get_trial(trial_id).value == expected
    assert storage.get_trial(trial_id).intermediate_values[1] == expected


def test_store_nan_intermediate_values() -> None:
    storage = get_storage()
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    trial_id = storage.create_new_trial(study_id)

    value = float("nan")
    storage.set_trial_intermediate_value(trial_id, 1, value)

    got_value = storage.get_trial(trial_id).intermediate_values[1]
    assert np.isnan(got_value)


def test_multithread_create_study() -> None:
    storage = get_storage()
    with ThreadPoolExecutor(10) as pool:
        for _ in range(10):
            pool.submit(
                optuna.create_study,
                storage=storage,
                study_name="test-multithread-create-study",
                load_if_exists=True,
            )


def test_multiprocess_run_optimize() -> None:
    n_workers = 8
    n_trials = 20
    storage = get_storage()
    try:
        optuna.delete_study(study_name=_STUDY_NAME, storage=storage)
    except KeyError:
        pass
    optuna.create_study(storage=storage, study_name=_STUDY_NAME)
    with ProcessPoolExecutor(n_workers) as pool:
        pool.map(run_optimize, *zip(*[[_STUDY_NAME, n_trials]] * n_workers))

    study = optuna.load_study(study_name=_STUDY_NAME, storage=storage)

    trials = study.trials
    assert len(trials) == n_workers * n_trials

    _check_trials(trials)


def test_pickle_storage() -> None:
    storage = get_storage()
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    storage.set_study_system_attr(study_id, "key", "pickle")

    restored_storage = pickle.loads(pickle.dumps(storage))

    storage_system_attrs = storage.get_study_system_attrs(study_id)
    restored_storage_system_attrs = restored_storage.get_study_system_attrs(study_id)
    assert storage_system_attrs == restored_storage_system_attrs == {"key": "pickle"}


@pytest.mark.parametrize("direction", [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE])
@pytest.mark.parametrize(
    "values",
    [
        [0.0, 1.0, 2.0],
        [0.0, float("inf"), 1.0],
        [0.0, float("-inf"), 1.0],
        [float("inf"), 0.0, 1.0, float("-inf")],
        [float("inf")],
        [float("-inf")],
    ],
)
def test_get_best_trial(direction: StudyDirection, values: Sequence[float]) -> None:
    storage = get_storage()
    study = optuna.create_study(direction=direction, storage=storage)
    study.add_trials(
        [optuna.create_trial(params={}, distributions={}, value=value) for value in values]
    )
    expected_value = max(values) if direction == StudyDirection.MAXIMIZE else min(values)
    assert study.best_value == expected_value
