from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import math
import os
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.storages import BaseStorage
from optuna.storages.journal import JournalRedisBackend
from optuna.study import StudyDirection
from optuna.trial import TrialState


_STUDY_NAME = "_test_multiprocess"


FLOAT_VALUES = (
    0,
    math.pi,
    sys.float_info.max,
    -sys.float_info.max,
    sys.float_info.min,
    -sys.float_info.min,
    float("inf"),
    -float("inf"),
    float("nan"),
)

FLOAT_ATTRS = {
    "zero": 0,
    "pi": math.pi,
    "max": sys.float_info.max,
    "negative max": -sys.float_info.max,
    "min": sys.float_info.min,
    "negative min": -sys.float_info.min,
    "inf": float("inf"),
    "negative inf": -float("inf"),
    "nan": float("nan"),
}


def is_equal_floats(a: float, b: float) -> bool:
    if math.isnan(a):
        return math.isnan(b)
    if math.isnan(b):
        return False
    return a == b


def is_equal_float_dicts(a: dict[str, float], b: dict[str, float]) -> bool:
    if a.keys() != b.keys():
        return False
    for key, value in a.items():
        if not is_equal_floats(value, b[key]):
            False
    return True


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


def test_set_and_get_study_user_attrs_for_floats() -> None:
    storage = get_storage()
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

    def check_set_and_get(key: str, value: Any) -> None:
        storage.set_study_user_attr(study_id, key, value)
        assert is_equal_floats(storage.get_study_user_attrs(study_id)[key], value)

    # Test setting value.
    for key, value in FLOAT_ATTRS.items():
        check_set_and_get(key, value)
    assert is_equal_float_dicts(storage.get_study_user_attrs(study_id), FLOAT_ATTRS)


def test_set_and_get_study_system_attrs_for_floats() -> None:
    storage = get_storage()
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])

    def check_set_and_get(key: str, value: Any) -> None:
        storage.set_study_system_attr(study_id, key, value)
        assert is_equal_floats(storage.get_study_system_attrs(study_id)[key], value)

    # Test setting value.
    for key, value in FLOAT_ATTRS.items():
        check_set_and_get(key, value)
    assert is_equal_float_dicts(storage.get_study_system_attrs(study_id), FLOAT_ATTRS)


def test_set_trial_state_values_for_floats() -> None:
    storage = get_storage()
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    for value in FLOAT_VALUES:
        if math.isnan(value):
            continue
        trial_id = storage.create_new_trial(study_id)
        storage.set_trial_state_values(trial_id, state=TrialState.COMPLETE, values=(value,))
        set_value = storage.get_trial(trial_id).value
        assert set_value is not None
        assert is_equal_floats(set_value, value)


def test_set_trial_param_for_floats() -> None:
    storage = get_storage()
    # Setup test across multiple studies and trials.
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    trial_id = storage.create_new_trial(study_id)

    for key, value in FLOAT_ATTRS.items():
        # MySQL cannot handle infinities.
        if not math.isfinite(value):
            continue
        param_name = "float_" + key
        float_dist = FloatDistribution(low=value, high=value)
        internal_repr = float_dist.to_internal_repr(value)
        storage.set_trial_param(trial_id, param_name, internal_repr, float_dist)
        assert is_equal_floats(storage.get_trial_param(trial_id, param_name), internal_repr)
        assert storage.get_trial(trial_id).distributions[param_name] == float_dist

    for key, value in FLOAT_ATTRS.items():
        param_name = "categorical_" + key
        categorical_dist = CategoricalDistribution(choices=(value,))
        internal_repr = categorical_dist.to_internal_repr(value)
        storage.set_trial_param(trial_id, param_name, internal_repr, categorical_dist)
        assert is_equal_floats(storage.get_trial_param(trial_id, param_name), internal_repr)
        assert storage.get_trial(trial_id).distributions[param_name] == categorical_dist


def test_set_trial_intermediate_value_for_floats() -> None:
    storage = get_storage()
    study_id = storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    trial_id = storage.create_new_trial(study_id)
    for i, value in enumerate(FLOAT_VALUES):
        storage.set_trial_intermediate_value(trial_id, i, value)
        assert is_equal_floats(storage.get_trial(trial_id).intermediate_values[i], value)


def test_set_trial_user_attr_for_floats() -> None:
    storage = get_storage()
    trial_id = storage.create_new_trial(
        storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    )

    def check_set_and_get(trial_id: int, key: str, value: Any) -> None:
        storage.set_trial_user_attr(trial_id, key, value)
        assert is_equal_floats(storage.get_trial(trial_id).user_attrs[key], value)

    # Test setting value.
    for key, value in FLOAT_ATTRS.items():
        check_set_and_get(trial_id, key, value)
    assert is_equal_float_dicts(storage.get_trial(trial_id).user_attrs, FLOAT_ATTRS)


def test_set_trial_system_attr_for_floats() -> None:
    storage = get_storage()
    trial_id = storage.create_new_trial(
        storage.create_new_study(directions=[StudyDirection.MINIMIZE])
    )

    def check_set_and_get(trial_id: int, key: str, value: Any) -> None:
        storage.set_trial_system_attr(trial_id, key, value)
        assert is_equal_floats(storage.get_trial(trial_id).system_attrs[key], value)

    # Test setting value.
    for key, value in FLOAT_ATTRS.items():
        check_set_and_get(trial_id, key, value)
    assert is_equal_float_dicts(storage.get_trial(trial_id).system_attrs, FLOAT_ATTRS)
