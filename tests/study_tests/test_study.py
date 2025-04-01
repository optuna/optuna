from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
import copy
import multiprocessing
import pickle
import platform
import threading
import time
from typing import Any
from typing import Callable as TypingCallable
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch
import uuid
import warnings

import _pytest.capture
import pytest

import optuna
from optuna import copy_study
from optuna import create_study
from optuna import create_trial
from optuna import delete_study
from optuna import distributions
from optuna import get_all_study_names
from optuna import get_all_study_summaries
from optuna import load_study
from optuna import logging
from optuna import Study
from optuna import Trial
from optuna import TrialPruned
from optuna.exceptions import DuplicatedStudyError
from optuna.exceptions import ExperimentalWarning
from optuna.study import StudyDirection
from optuna.study._constrained_optimization import _CONSTRAINTS_KEY
from optuna.study.study import _SYSTEM_ATTR_METRIC_NAMES
from optuna.testing.objectives import fail_objective
from optuna.testing.storages import STORAGE_MODES
from optuna.testing.storages import StorageSupplier
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


CallbackFuncType = TypingCallable[[Study, FrozenTrial], None]


def func(trial: Trial) -> float:
    x = trial.suggest_float("x", -10.0, 10.0)
    y = trial.suggest_float("y", 20, 30, log=True)
    z = trial.suggest_categorical("z", (-1.0, 1.0))
    return (x - 2) ** 2 + (y - 25) ** 2 + z


class Func:
    def __init__(self, sleep_sec: float | None = None) -> None:
        self.n_calls = 0
        self.sleep_sec = sleep_sec
        self.lock = threading.Lock()

    def __call__(self, trial: Trial) -> float:
        with self.lock:
            self.n_calls += 1

        # Sleep for testing parallelism.
        if self.sleep_sec is not None:
            time.sleep(self.sleep_sec)

        value = func(trial)
        check_params(trial.params)
        return value


def check_params(params: dict[str, Any]) -> None:
    assert sorted(params.keys()) == ["x", "y", "z"]


def check_value(value: float | None) -> None:
    assert isinstance(value, float)
    assert -1.0 <= value <= 12.0**2 + 5.0**2 + 1.0


def check_frozen_trial(frozen_trial: FrozenTrial) -> None:
    if frozen_trial.state == TrialState.COMPLETE:
        check_params(frozen_trial.params)
        check_value(frozen_trial.value)


def check_study(study: Study) -> None:
    for trial in study.trials:
        check_frozen_trial(trial)

    assert not study._is_multi_objective()

    complete_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
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


def stop_objective(threshold_number: int) -> Callable[[Trial], float]:
    def objective(trial: Trial) -> float:
        if trial.number >= threshold_number:
            trial.study.stop()

        return trial.number

    return objective


def test_optimize_trivial_in_memory_new() -> None:
    study = create_study()
    study.optimize(func, n_trials=10)
    check_study(study)


def test_optimize_trivial_in_memory_resume() -> None:
    study = create_study()
    study.optimize(func, n_trials=10)
    study.optimize(func, n_trials=10)
    check_study(study)


def test_optimize_trivial_rdb_resume_study() -> None:
    study = create_study(storage="sqlite:///:memory:")
    study.optimize(func, n_trials=10)
    check_study(study)


def test_optimize_with_direction() -> None:
    study = create_study(direction="minimize")
    study.optimize(func, n_trials=10)
    assert study.direction == StudyDirection.MINIMIZE
    check_study(study)

    study = create_study(direction="maximize")
    study.optimize(func, n_trials=10)
    assert study.direction == StudyDirection.MAXIMIZE
    check_study(study)

    with pytest.raises(ValueError):
        create_study(direction="test")

    with pytest.raises(ValueError):
        create_study(direction=["maximize", "minimize"])  # type: ignore [arg-type]

    with pytest.raises(ValueError):
        create_study(directions="minimize")


@pytest.mark.parametrize("n_trials", (0, 1, 20))
@pytest.mark.parametrize("n_jobs", (1, 2, -1))
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_optimize_parallel(n_trials: int, n_jobs: int, storage_mode: str) -> None:
    f = Func()

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(f, n_trials=n_trials, n_jobs=n_jobs)
        assert f.n_calls == len(study.trials) == n_trials
        check_study(study)


def test_optimize_with_thread_pool_executor() -> None:
    def objective(t: Trial) -> float:
        return t.suggest_float("x", -10, 10)

    study = create_study()
    with ThreadPoolExecutor(max_workers=5) as pool:
        for _ in range(10):
            pool.submit(study.optimize, objective, n_trials=10)
    assert len(study.trials) == 100


@pytest.mark.parametrize("n_trials", (0, 1, 20, None))
@pytest.mark.parametrize("n_jobs", (1, 2, -1))
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_optimize_parallel_timeout(n_trials: int, n_jobs: int, storage_mode: str) -> None:
    sleep_sec = 0.1
    timeout_sec = 1.0
    f = Func(sleep_sec=sleep_sec)

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(f, n_trials=n_trials, n_jobs=n_jobs, timeout=timeout_sec)

        assert f.n_calls == len(study.trials)

        if n_trials is not None:
            assert f.n_calls <= n_trials

        # A thread can process at most (timeout_sec / sleep_sec + 1) trials.
        n_jobs_actual = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        max_calls = (timeout_sec / sleep_sec + 1) * n_jobs_actual
        assert f.n_calls <= max_calls

        check_study(study)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_optimize_with_catch(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        # Test default exceptions.
        with pytest.raises(ValueError):
            study.optimize(fail_objective, n_trials=20)
        assert len(study.trials) == 1
        assert all(trial.state == TrialState.FAIL for trial in study.trials)

        # Test acceptable exception.
        study.optimize(fail_objective, n_trials=20, catch=(ValueError,))
        assert len(study.trials) == 21
        assert all(trial.state == TrialState.FAIL for trial in study.trials)

        # Test trial with unacceptable exception.
        with pytest.raises(ValueError):
            study.optimize(fail_objective, n_trials=20, catch=(ArithmeticError,))
        assert len(study.trials) == 22
        assert all(trial.state == TrialState.FAIL for trial in study.trials)


@pytest.mark.parametrize("catch", [ValueError, (ValueError,), [ValueError], {ValueError}])
def test_optimize_with_catch_valid_type(catch: Any) -> None:
    study = create_study()
    study.optimize(fail_objective, n_trials=20, catch=catch)


@pytest.mark.parametrize("catch", [None, 1])
def test_optimize_with_catch_invalid_type(catch: Any) -> None:
    study = create_study()

    with pytest.raises(TypeError):
        study.optimize(fail_objective, n_trials=20, catch=catch)


@pytest.mark.parametrize("n_jobs", (2, -1))
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_optimize_with_reseeding(n_jobs: int, storage_mode: str) -> None:
    f = Func()

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        sampler = study.sampler
        with patch.object(sampler, "reseed_rng", wraps=sampler.reseed_rng) as mock_object:
            study.optimize(f, n_trials=1, n_jobs=2)
            assert mock_object.call_count == 1


def test_call_another_study_optimize_in_optimize() -> None:
    def inner_objective(t: Trial) -> float:
        return t.suggest_float("x", -10, 10)

    def objective(t: Trial) -> float:
        inner_study = create_study()
        inner_study.enqueue_trial({"x": t.suggest_int("initial_point", -10, 10)})
        inner_study.optimize(inner_objective, n_trials=10)
        return inner_study.best_value

    study = create_study()
    study.optimize(objective, n_trials=10)
    assert len(study.trials) == 10


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_study_set_and_get_user_attrs(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        study.set_user_attr("dataset", "MNIST")
        assert study.user_attrs["dataset"] == "MNIST"


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_trial_set_and_get_user_attrs(storage_mode: str) -> None:
    def f(trial: Trial) -> float:
        trial.set_user_attr("train_accuracy", 1)
        assert trial.user_attrs["train_accuracy"] == 1
        return 0.0

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(f, n_trials=1)
        frozen_trial = study.trials[0]
        assert frozen_trial.user_attrs["train_accuracy"] == 1


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize("include_best_trial", [True, False])
def test_get_all_study_summaries(storage_mode: str, include_best_trial: bool) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(func, n_trials=5)

        summaries = get_all_study_summaries(study._storage, include_best_trial)
        summary = [s for s in summaries if s._study_id == study._study_id][0]

        assert summary.study_name == study.study_name
        assert summary.n_trials == 5
        if include_best_trial:
            assert summary.best_trial is not None
        else:
            assert summary.best_trial is None


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_all_study_summaries_with_no_trials(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        summaries = get_all_study_summaries(study._storage)
        summary = [s for s in summaries if s._study_id == study._study_id][0]

        assert summary.study_name == study.study_name
        assert summary.n_trials == 0
        assert summary.datetime_start is None


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_all_study_names(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        n_studies = 5

        studies = [create_study(storage=storage) for _ in range(n_studies)]
        study_names = get_all_study_names(storage)

        assert len(study_names) == n_studies
        for study, study_name in zip(studies, study_names):
            assert study_name == study.study_name


def test_study_pickle() -> None:
    study_1 = create_study()
    study_1.optimize(func, n_trials=10)
    check_study(study_1)
    assert len(study_1.trials) == 10
    dumped_bytes = pickle.dumps(study_1)

    study_2 = pickle.loads(dumped_bytes)
    check_study(study_2)
    assert len(study_2.trials) == 10

    study_2.optimize(func, n_trials=10)
    check_study(study_2)
    assert len(study_2.trials) == 20


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_create_study(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        # Test creating a new study.
        study = create_study(storage=storage, load_if_exists=False)

        # Test `load_if_exists=True` with existing study.
        create_study(study_name=study.study_name, storage=storage, load_if_exists=True)

        with pytest.raises(DuplicatedStudyError):
            create_study(study_name=study.study_name, storage=storage, load_if_exists=False)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_load_study(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        if storage is None:
            # :class:`~optuna.storages.InMemoryStorage` can not be used with `load_study` function.
            return

        study_name = str(uuid.uuid4())

        with pytest.raises(KeyError):
            # Test loading an unexisting study.
            load_study(study_name=study_name, storage=storage)

        # Create a new study.
        created_study = create_study(study_name=study_name, storage=storage)

        # Test loading an existing study.
        loaded_study = load_study(study_name=study_name, storage=storage)
        assert created_study._study_id == loaded_study._study_id


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_load_study_study_name_none(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        if storage is None:
            # :class:`~optuna.storages.InMemoryStorage` can not be used with `load_study` function.
            return

        study_name = str(uuid.uuid4())

        _ = create_study(study_name=study_name, storage=storage)

        loaded_study = load_study(study_name=None, storage=storage)

        assert loaded_study.study_name == study_name

        study_name = str(uuid.uuid4())

        _ = create_study(study_name=study_name, storage=storage)

        # Ambiguous study.
        with pytest.raises(ValueError):
            load_study(study_name=None, storage=storage)


def test_load_study_default_sampler() -> None:
    storage = optuna.storages.InMemoryStorage()

    # Single-objective
    study_name = str(uuid.uuid4())
    create_study(storage=storage, study_name=study_name)
    loaded_study = load_study(study_name=study_name, storage=storage)
    assert isinstance(loaded_study.sampler, optuna.samplers.TPESampler)

    # Multi-objective
    study_name = str(uuid.uuid4())
    create_study(storage=storage, study_name=study_name, directions=["minimize", "maximize"])
    loaded_study = load_study(study_name=study_name, storage=storage)
    assert isinstance(loaded_study.sampler, optuna.samplers.NSGAIISampler)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_delete_study(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        # Test deleting a non-existing study.
        with pytest.raises(KeyError):
            delete_study(study_name="invalid-study-name", storage=storage)

        # Test deleting an existing study.
        study = create_study(storage=storage, load_if_exists=False)
        delete_study(study_name=study.study_name, storage=storage)

        # Test failed to delete the study which is already deleted.
        with pytest.raises(KeyError):
            delete_study(study_name=study.study_name, storage=storage)


@pytest.mark.parametrize("from_storage_mode", STORAGE_MODES)
@pytest.mark.parametrize("to_storage_mode", STORAGE_MODES)
def test_copy_study(from_storage_mode: str, to_storage_mode: str) -> None:
    with StorageSupplier(from_storage_mode) as from_storage, StorageSupplier(
        to_storage_mode
    ) as to_storage:
        from_study = create_study(storage=from_storage, directions=["maximize", "minimize"])
        from_study._storage.set_study_system_attr(from_study._study_id, "foo", "bar")
        from_study.set_user_attr("baz", "qux")
        from_study.optimize(
            lambda t: (t.suggest_float("x0", 0, 1), t.suggest_float("x1", 0, 1)), n_trials=3
        )

        copy_study(
            from_study_name=from_study.study_name,
            from_storage=from_storage,
            to_storage=to_storage,
        )

        to_study = load_study(study_name=from_study.study_name, storage=to_storage)

        assert to_study.study_name == from_study.study_name
        assert to_study.directions == from_study.directions
        to_study_system_attrs = to_study._storage.get_study_system_attrs(to_study._study_id)
        from_study_system_attrs = from_study._storage.get_study_system_attrs(from_study._study_id)
        assert to_study_system_attrs == from_study_system_attrs
        assert to_study.user_attrs == from_study.user_attrs
        assert len(to_study.trials) == len(from_study.trials)


@pytest.mark.parametrize("from_storage_mode", STORAGE_MODES)
@pytest.mark.parametrize("to_storage_mode", STORAGE_MODES)
def test_copy_study_to_study_name(from_storage_mode: str, to_storage_mode: str) -> None:
    with StorageSupplier(from_storage_mode) as from_storage, StorageSupplier(
        to_storage_mode
    ) as to_storage:
        from_study = create_study(study_name="foo", storage=from_storage)
        _ = create_study(study_name="foo", storage=to_storage)

        with pytest.raises(DuplicatedStudyError):
            copy_study(
                from_study_name=from_study.study_name,
                from_storage=from_storage,
                to_storage=to_storage,
            )

        copy_study(
            from_study_name=from_study.study_name,
            from_storage=from_storage,
            to_storage=to_storage,
            to_study_name="bar",
        )

        _ = load_study(study_name="bar", storage=to_storage)


def test_nested_optimization() -> None:
    def objective(trial: Trial) -> float:
        with pytest.raises(RuntimeError):
            trial.study.optimize(lambda _: 0.0, n_trials=1)

        return 1.0

    study = create_study()
    study.optimize(objective, n_trials=10, catch=())


def test_stop_in_objective() -> None:
    # Test stopping the optimization: it should stop once the trial number reaches 4.
    study = create_study()
    study.optimize(stop_objective(4), n_trials=10)
    assert len(study.trials) == 5

    # Test calling `optimize` again: it should stop once the trial number reaches 11.
    study.optimize(stop_objective(11), n_trials=10)
    assert len(study.trials) == 12


def test_stop_in_callback() -> None:
    def callback(study: Study, trial: FrozenTrial) -> None:
        if trial.number >= 4:
            study.stop()

    # Test stopping the optimization inside a callback.
    study = create_study()
    study.optimize(lambda _: 1.0, n_trials=10, callbacks=[callback])
    assert len(study.trials) == 5


def test_stop_n_jobs() -> None:
    def callback(study: Study, trial: FrozenTrial) -> None:
        if trial.number >= 4:
            study.stop()

    study = create_study()
    study.optimize(lambda _: 1.0, n_trials=None, callbacks=[callback], n_jobs=2)
    assert 5 <= len(study.trials) <= 6


def test_stop_outside_optimize() -> None:
    # Test stopping outside the optimization: it should raise `RuntimeError`.
    study = create_study()
    with pytest.raises(RuntimeError):
        study.stop()

    # Test calling `optimize` after the `RuntimeError` is caught.
    study.optimize(lambda _: 1.0, n_trials=1)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_add_trial(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        assert len(study.trials) == 0

        trial = create_trial(value=0.8)
        study.add_trial(trial)
        assert len(study.trials) == 1
        assert study.trials[0].number == 0
        assert study.best_value == 0.8


def test_add_trial_invalid_values_length() -> None:
    study = create_study()
    trial = create_trial(values=[0, 0])
    with pytest.raises(ValueError):
        study.add_trial(trial)

    study = create_study(directions=["minimize", "minimize"])
    trial = create_trial(value=0)
    with pytest.raises(ValueError):
        study.add_trial(trial)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_add_trials(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        assert len(study.trials) == 0

        study.add_trials([])
        assert len(study.trials) == 0

        trials = [create_trial(value=i) for i in range(3)]
        study.add_trials(trials)
        assert len(study.trials) == 3
        for i, trial in enumerate(study.trials):
            assert trial.number == i
            assert trial.value == i

        other_study = create_study(storage=storage)
        other_study.add_trials(study.trials)
        assert len(other_study.trials) == 3
        for i, trial in enumerate(other_study.trials):
            assert trial.number == i
            assert trial.value == i


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_enqueue_trial_properly_sets_param_values(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        assert len(study.trials) == 0

        study.enqueue_trial(params={"x": -5, "y": 5})
        study.enqueue_trial(params={"x": -1, "y": 0})

        def objective(trial: Trial) -> float:
            x = trial.suggest_int("x", -10, 10)
            y = trial.suggest_int("y", -10, 10)
            return x**2 + y**2

        study.optimize(objective, n_trials=2)
        t0 = study.trials[0]
        assert t0.params["x"] == -5
        assert t0.params["y"] == 5

        t1 = study.trials[1]
        assert t1.params["x"] == -1
        assert t1.params["y"] == 0


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_enqueue_trial_with_unfixed_parameters(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        assert len(study.trials) == 0

        study.enqueue_trial(params={"x": -5})

        def objective(trial: Trial) -> float:
            x = trial.suggest_int("x", -10, 10)
            y = trial.suggest_int("y", -10, 10)
            return x**2 + y**2

        study.optimize(objective, n_trials=1)
        t = study.trials[0]
        assert t.params["x"] == -5
        assert -10 <= t.params["y"] <= 10


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_enqueue_trial_properly_sets_user_attr(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        assert len(study.trials) == 0

        study.enqueue_trial(params={"x": -5, "y": 5}, user_attrs={"is_optimal": False})
        study.enqueue_trial(params={"x": 0, "y": 0}, user_attrs={"is_optimal": True})

        def objective(trial: Trial) -> float:
            x = trial.suggest_int("x", -10, 10)
            y = trial.suggest_int("y", -10, 10)
            return x**2 + y**2

        study.optimize(objective, n_trials=2)
        t0 = study.trials[0]
        assert t0.user_attrs == {"is_optimal": False}

        t1 = study.trials[1]
        assert t1.user_attrs == {"is_optimal": True}


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_enqueue_trial_with_non_dict_parameters(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        assert len(study.trials) == 0

        with pytest.raises(TypeError):
            study.enqueue_trial(params=[17, 12])  # type: ignore[arg-type]


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_enqueue_trial_with_out_of_range_parameters(storage_mode: str) -> None:
    fixed_value = 11

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        assert len(study.trials) == 0

        study.enqueue_trial(params={"x": fixed_value})

        def objective(trial: Trial) -> float:
            return trial.suggest_int("x", -10, 10)

        with pytest.warns(UserWarning):
            study.optimize(objective, n_trials=1)
        t = study.trials[0]
        assert t.params["x"] == fixed_value

    # Internal logic might differ when distribution contains a single element.
    # Test it explicitly.
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        assert len(study.trials) == 0

        study.enqueue_trial(params={"x": fixed_value})

        def objective(trial: Trial) -> float:
            return trial.suggest_int("x", 1, 1)  # Single element.

        with pytest.warns(UserWarning):
            study.optimize(objective, n_trials=1)
        t = study.trials[0]
        assert t.params["x"] == fixed_value


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_enqueue_trial_skips_existing_finished(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        assert len(study.trials) == 0

        def objective(trial: Trial) -> float:
            x = trial.suggest_int("x", -10, 10)
            y = trial.suggest_int("y", -10, 10)
            return x**2 + y**2

        study.enqueue_trial({"x": -5, "y": 5})
        study.optimize(objective, n_trials=1)

        t0 = study.trials[0]
        assert t0.params["x"] == -5
        assert t0.params["y"] == 5

        before_enqueue = len(study.trials)
        study.enqueue_trial({"x": -5, "y": 5}, skip_if_exists=True)
        after_enqueue = len(study.trials)
        assert before_enqueue == after_enqueue


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_enqueue_trial_skips_existing_waiting(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        assert len(study.trials) == 0

        def objective(trial: Trial) -> float:
            x = trial.suggest_int("x", -10, 10)
            y = trial.suggest_int("y", -10, 10)
            return x**2 + y**2

        study.enqueue_trial({"x": -5, "y": 5})
        before_enqueue = len(study.trials)
        study.enqueue_trial({"x": -5, "y": 5}, skip_if_exists=True)
        after_enqueue = len(study.trials)
        assert before_enqueue == after_enqueue

        study.optimize(objective, n_trials=1)
        t0 = study.trials[0]
        assert t0.params["x"] == -5
        assert t0.params["y"] == 5


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize(
    "new_params", [{"x": -5, "y": 5, "z": 5}, {"x": -5}, {"x": -5, "z": 5}, {"x": -5, "y": 6}]
)
def test_enqueue_trial_skip_existing_allows_unfixed(
    storage_mode: str, new_params: dict[str, int]
) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        assert len(study.trials) == 0

        def objective(trial: Trial) -> float:
            x = trial.suggest_int("x", -10, 10)
            y = trial.suggest_int("y", -10, 10)
            if trial.number == 1:
                z = trial.suggest_int("z", -10, 10)
                return x**2 + y**2 + z**2
            return x**2 + y**2

        study.enqueue_trial({"x": -5, "y": 5})
        study.optimize(objective, n_trials=1)
        t0 = study.trials[0]
        assert t0.params["x"] == -5
        assert t0.params["y"] == 5

        study.enqueue_trial(new_params, skip_if_exists=True)
        study.optimize(objective, n_trials=1)

        unfixed_params = {"x", "y", "z"} - set(new_params)
        t1 = study.trials[1]
        assert all(t1.params[k] == new_params[k] for k in new_params)
        assert all(-10 <= t1.params[k] <= 10 for k in unfixed_params)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize(
    "param", ["foo", 1, 1.1, 1e17, 1e-17, float("inf"), float("-inf"), float("nan"), None]
)
def test_enqueue_trial_skip_existing_handles_common_types(storage_mode: str, param: Any) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.enqueue_trial({"x": param})
        before_enqueue = len(study.trials)
        study.enqueue_trial({"x": param}, skip_if_exists=True)
        after_enqueue = len(study.trials)
        assert before_enqueue == after_enqueue


@patch("optuna.study._optimize.gc.collect")
def test_optimize_with_gc(collect_mock: Mock) -> None:
    study = create_study()
    study.optimize(func, n_trials=10, gc_after_trial=True)
    check_study(study)
    assert collect_mock.call_count == 10


@patch("optuna.study._optimize.gc.collect")
def test_optimize_without_gc(collect_mock: Mock) -> None:
    study = create_study()
    study.optimize(func, n_trials=10, gc_after_trial=False)
    check_study(study)
    assert collect_mock.call_count == 0


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_optimize_with_progbar(n_jobs: int, capsys: _pytest.capture.CaptureFixture) -> None:
    study = create_study()
    study.optimize(lambda _: 1.0, n_trials=10, n_jobs=n_jobs, show_progress_bar=True)
    _, err = capsys.readouterr()

    # Search for progress bar elements in stderr.
    assert "Best trial: 0" in err
    assert "Best value: 1" in err
    assert "10/10" in err
    if platform.system() != "Windows":
        # Skip this assertion because the progress bar sometimes stops at 99% on Windows.
        assert "100%" in err


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_optimize_without_progbar(n_jobs: int, capsys: _pytest.capture.CaptureFixture) -> None:
    study = create_study()
    study.optimize(lambda _: 1.0, n_trials=10, n_jobs=n_jobs)
    _, err = capsys.readouterr()

    assert "Best trial: 0" not in err
    assert "Best value: 1" not in err
    assert "10/10" not in err
    if platform.system() != "Windows":
        # Skip this assertion because the progress bar sometimes stops at 99% on Windows.
        assert "100%" not in err


def test_optimize_with_progbar_timeout(capsys: _pytest.capture.CaptureFixture) -> None:
    study = create_study()
    study.optimize(lambda _: 1.0, timeout=2.0, show_progress_bar=True)
    _, err = capsys.readouterr()

    assert "Best trial: 0" in err
    assert "Best value: 1" in err
    assert "00:02/00:02" in err
    if platform.system() != "Windows":
        # Skip this assertion because the progress bar sometimes stops at 99% on Windows.
        assert "100%" in err


def test_optimize_with_progbar_parallel_timeout(capsys: _pytest.capture.CaptureFixture) -> None:
    study = create_study()
    with pytest.warns(
        UserWarning, match="The timeout-based progress bar is not supported with n_jobs != 1."
    ):
        study.optimize(lambda _: 1.0, timeout=2.0, show_progress_bar=True, n_jobs=2)
    _, err = capsys.readouterr()

    # Testing for a character that forms progress bar borders.
    assert "|" not in err


@pytest.mark.parametrize(
    "timeout,expected",
    [
        (59.0, "/00:59"),
        (60.0, "/01:00"),
        (60.0 * 60, "/1:00:00"),
        (60.0 * 60 * 24, "/24:00:00"),
        (60.0 * 60 * 24 * 10, "/240:00:00"),
    ],
)
def test_optimize_with_progbar_timeout_formats(
    timeout: float, expected: str, capsys: _pytest.capture.CaptureFixture
) -> None:
    study = create_study()
    study.optimize(stop_objective(5), timeout=timeout, show_progress_bar=True)
    _, err = capsys.readouterr()
    assert expected in err


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_optimize_without_progbar_timeout(
    n_jobs: int, capsys: _pytest.capture.CaptureFixture
) -> None:
    study = create_study()
    study.optimize(lambda _: 1.0, timeout=2.0, n_jobs=n_jobs)
    _, err = capsys.readouterr()

    assert "Best trial: 0" not in err
    assert "Best value: 1.0" not in err
    assert "00:02/00:02" not in err
    if platform.system() != "Windows":
        # Skip this assertion because the progress bar sometimes stops at 99% on Windows.
        assert "100%" not in err


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_optimize_progbar_n_trials_prioritized(
    n_jobs: int, capsys: _pytest.capture.CaptureFixture
) -> None:
    study = create_study()
    study.optimize(lambda _: 1.0, n_trials=10, n_jobs=n_jobs, timeout=10.0, show_progress_bar=True)
    _, err = capsys.readouterr()

    assert "Best trial: 0" in err
    assert "Best value: 1" in err
    assert "10/10" in err
    if platform.system() != "Windows":
        # Skip this assertion because the progress bar sometimes stops at 99% on Windows.
        assert "100%" in err
    assert "it" in err


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_optimize_without_progbar_n_trials_prioritized(
    n_jobs: int, capsys: _pytest.capture.CaptureFixture
) -> None:
    study = create_study()
    study.optimize(lambda _: 1.0, n_trials=10, n_jobs=n_jobs, timeout=10.0)
    _, err = capsys.readouterr()

    # Testing for a character that forms progress bar borders.
    assert "|" not in err


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_optimize_progbar_no_constraints(
    n_jobs: int, capsys: _pytest.capture.CaptureFixture
) -> None:
    study = create_study()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        study.optimize(stop_objective(5), n_jobs=n_jobs, show_progress_bar=True)
    _, err = capsys.readouterr()

    # We can't simply test if stderr is empty, since we're not sure
    # what else could write to it. Instead, we are testing for a character
    # that forms progress bar borders.
    assert "|" not in err


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_optimize_without_progbar_no_constraints(
    n_jobs: int, capsys: _pytest.capture.CaptureFixture
) -> None:
    study = create_study()
    study.optimize(stop_objective(5), n_jobs=n_jobs)
    _, err = capsys.readouterr()

    # Testing for a character that forms progress bar borders.
    assert "|" not in err


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_callbacks(n_jobs: int) -> None:
    lock = threading.Lock()

    def with_lock(f: CallbackFuncType) -> CallbackFuncType:
        def callback(study: Study, trial: FrozenTrial) -> None:
            with lock:
                f(study, trial)

        return callback

    study = create_study()

    def objective(trial: Trial) -> float:
        return trial.suggest_int("x", 1, 1)

    # Empty callback list.
    study.optimize(objective, callbacks=[], n_trials=10, n_jobs=n_jobs)

    # One callback.
    values = []
    callbacks = [with_lock(lambda study, trial: values.append(trial.value))]
    study.optimize(objective, callbacks=callbacks, n_trials=10, n_jobs=n_jobs)
    assert values == [1] * 10

    # Two callbacks.
    values = []
    params = []
    callbacks = [
        with_lock(lambda study, trial: values.append(trial.value)),
        with_lock(lambda study, trial: params.append(trial.params)),
    ]
    study.optimize(objective, callbacks=callbacks, n_trials=10, n_jobs=n_jobs)
    assert values == [1] * 10
    assert params == [{"x": 1}] * 10

    # If a trial is failed with an exception and the exception is caught by the study,
    # callbacks are invoked.
    states = []
    callbacks = [with_lock(lambda study, trial: states.append(trial.state))]
    study.optimize(
        lambda t: 1 / 0,
        callbacks=callbacks,
        n_trials=10,
        n_jobs=n_jobs,
        catch=(ZeroDivisionError,),
    )
    assert states == [TrialState.FAIL] * 10

    # If a trial is failed with an exception and the exception isn't caught by the study,
    # callbacks aren't invoked.
    states = []
    callbacks = [with_lock(lambda study, trial: states.append(trial.state))]
    with pytest.raises(ZeroDivisionError):
        study.optimize(lambda t: 1 / 0, callbacks=callbacks, n_trials=10, n_jobs=n_jobs, catch=())
    assert states == []


def test_optimize_infinite_budget_progbar() -> None:
    def terminate_study(study: Study, trial: FrozenTrial) -> None:
        study.stop()

    study = create_study()

    with pytest.warns(UserWarning):
        study.optimize(
            func, n_trials=None, timeout=None, show_progress_bar=True, callbacks=[terminate_study]
        )


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_trials(storage_mode: str) -> None:
    if storage_mode in ("grpc_rdb", "grpc_journal_file"):
        pytest.skip("gRPC storage doesn't use `copy.deepcopy`.")

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(lambda t: t.suggest_int("x", 1, 5), n_trials=5)

        with patch("copy.deepcopy", wraps=copy.deepcopy) as mock_object:
            trials0 = study.get_trials(deepcopy=False)
            assert mock_object.call_count == 0
            assert len(trials0) == 5

            trials1 = study.get_trials(deepcopy=True)
            assert mock_object.call_count > 0
            assert trials0 == trials1

            # `study.trials` is equivalent to `study.get_trials(deepcopy=True)`.
            old_count = mock_object.call_count
            trials2 = study.trials
            assert mock_object.call_count > old_count
            assert trials0 == trials2


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_trials_state_option(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        def objective(trial: Trial) -> float:
            if trial.number == 0:
                return 0.0  # TrialState.COMPLETE.
            elif trial.number == 1:
                return 0.0  # TrialState.COMPLETE.
            elif trial.number == 2:
                raise TrialPruned  # TrialState.PRUNED.
            else:
                assert False

        study.optimize(objective, n_trials=3)

        trials = study.get_trials(states=None)
        assert len(trials) == 3

        trials = study.get_trials(states=(TrialState.COMPLETE,))
        assert len(trials) == 2
        assert all(t.state == TrialState.COMPLETE for t in trials)

        trials = study.get_trials(states=(TrialState.COMPLETE, TrialState.PRUNED))
        assert len(trials) == 3
        assert all(t.state in (TrialState.COMPLETE, TrialState.PRUNED) for t in trials)

        trials = study.get_trials(states=())
        assert len(trials) == 0

        other_states = [
            s for s in list(TrialState) if s != TrialState.COMPLETE and s != TrialState.PRUNED
        ]
        for s in other_states:
            trials = study.get_trials(states=(s,))
            assert len(trials) == 0


def test_log_completed_trial(capsys: _pytest.capture.CaptureFixture) -> None:
    # We need to reconstruct our default handler to properly capture stderr.
    logging._reset_library_root_logger()
    logging.set_verbosity(logging.INFO)

    study = create_study()
    study.optimize(lambda _: 1.0, n_trials=1)
    _, err = capsys.readouterr()
    assert "Trial 0" in err

    logging.set_verbosity(logging.WARNING)
    study.optimize(lambda _: 1.0, n_trials=1)
    _, err = capsys.readouterr()
    assert "Trial 1" not in err

    logging.set_verbosity(logging.DEBUG)
    study.optimize(lambda _: 1.0, n_trials=1)
    _, err = capsys.readouterr()
    assert "Trial 2" in err


def test_log_completed_trial_skip_storage_access() -> None:
    study = create_study()

    # Create a trial to retrieve it as the `study.best_trial`.
    study.optimize(lambda _: 0.0, n_trials=1)
    frozen_trial = study.best_trial

    storage = study._storage

    with patch.object(storage, "get_best_trial", wraps=storage.get_best_trial) as mock_object:
        study._log_completed_trial(frozen_trial)
        assert mock_object.call_count == 1

    logging.set_verbosity(logging.WARNING)
    with patch.object(storage, "get_best_trial", wraps=storage.get_best_trial) as mock_object:
        study._log_completed_trial(frozen_trial)
        assert mock_object.call_count == 0

    logging.set_verbosity(logging.DEBUG)
    with patch.object(storage, "get_best_trial", wraps=storage.get_best_trial) as mock_object:
        study._log_completed_trial(frozen_trial)
        assert mock_object.call_count == 1


def test_create_study_with_multi_objectives() -> None:
    study = create_study(directions=["maximize"])
    assert study.direction == StudyDirection.MAXIMIZE
    assert not study._is_multi_objective()

    study = create_study(directions=["maximize", "minimize"])
    assert study.directions == [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]
    assert study._is_multi_objective()

    with pytest.raises(ValueError):
        # Empty `direction` isn't allowed.
        _ = create_study(directions=[])

    with pytest.raises(ValueError):
        _ = create_study(direction="minimize", directions=["maximize"])

    with pytest.raises(ValueError):
        _ = create_study(direction="minimize", directions=[])


def test_create_study_with_direction_object() -> None:
    study = create_study(direction=StudyDirection.MAXIMIZE)
    assert study.direction == StudyDirection.MAXIMIZE

    study = create_study(directions=[StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE])
    assert study.directions == [StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]


@pytest.mark.parametrize("n_objectives", [2, 3])
def test_optimize_with_multi_objectives(n_objectives: int) -> None:
    directions = ["minimize" for _ in range(n_objectives)]
    study = create_study(directions=directions)

    def objective(trial: Trial) -> list[float]:
        return [trial.suggest_float("v{}".format(i), 0, 5) for i in range(n_objectives)]

    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 10

    for trial in study.trials:
        assert trial.values
        assert len(trial.values) == n_objectives


@pytest.mark.parametrize("direction", [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
def test_best_trial_constrained_optimization(direction: StudyDirection) -> None:
    study = create_study(direction=direction)
    storage = study._storage

    with pytest.raises(ValueError):
        # No trials.
        study.best_trial

    trial = study.ask()
    storage.set_trial_system_attr(trial._trial_id, _CONSTRAINTS_KEY, [1])
    study.tell(trial, 0)
    with pytest.raises(ValueError):
        # No feasible trials.
        study.best_trial

    trial = study.ask()
    storage.set_trial_system_attr(trial._trial_id, _CONSTRAINTS_KEY, [0])
    study.tell(trial, 0)
    assert study.best_trial.number == 1

    trial = study.ask()
    storage.set_trial_system_attr(trial._trial_id, _CONSTRAINTS_KEY, [1])
    study.tell(trial, -1 if direction == StudyDirection.MINIMIZE else 1)
    assert study.best_trial.number == 1

    trial = study.ask()
    storage.set_trial_system_attr(trial._trial_id, _CONSTRAINTS_KEY, [0])
    study.tell(trial, -1 if direction == StudyDirection.MINIMIZE else 1)
    assert study.best_trial.number == 3


def test_best_trials() -> None:
    study = create_study(directions=["minimize", "maximize"])
    study.optimize(lambda t: [2, 2], n_trials=1)
    study.optimize(lambda t: [1, 1], n_trials=1)
    study.optimize(lambda t: [3, 1], n_trials=1)
    assert {tuple(t.values) for t in study.best_trials} == {(1, 1), (2, 2)}


def test_best_trials_constrained_optimization() -> None:
    study = create_study(directions=["minimize", "maximize"])
    storage = study._storage

    assert study.best_trials == []

    trial = study.ask()
    storage.set_trial_system_attr(trial._trial_id, _CONSTRAINTS_KEY, [1])
    study.tell(trial, [0, 0])
    assert study.best_trials == []

    trial = study.ask()
    storage.set_trial_system_attr(trial._trial_id, _CONSTRAINTS_KEY, [0])
    study.tell(trial, [0, 0])
    assert study.best_trials == [study.trials[1]]

    trial = study.ask()
    storage.set_trial_system_attr(trial._trial_id, _CONSTRAINTS_KEY, [1])
    study.tell(trial, [-1, 1])
    assert study.best_trials == [study.trials[1]]

    trial = study.ask()
    storage.set_trial_system_attr(trial._trial_id, _CONSTRAINTS_KEY, [0])
    study.tell(trial, [1, 1])
    assert {t.number for t in study.best_trials} == {1, 3}


def test_wrong_n_objectives() -> None:
    n_objectives = 2
    directions = ["minimize" for _ in range(n_objectives)]
    study = create_study(directions=directions)

    def objective(trial: Trial) -> list[float]:
        return [trial.suggest_float("v{}".format(i), 0, 5) for i in range(n_objectives + 1)]

    study.optimize(objective, n_trials=10)

    for trial in study.trials:
        assert trial.state is TrialState.FAIL


def test_ask() -> None:
    study = create_study()

    trial = study.ask()
    assert isinstance(trial, Trial)


def test_ask_enqueue_trial() -> None:
    study = create_study()

    study.enqueue_trial({"x": 0.5}, user_attrs={"memo": "this is memo"})

    trial = study.ask()
    assert trial.suggest_float("x", 0, 1) == 0.5
    assert trial.user_attrs == {"memo": "this is memo"}


def test_ask_fixed_search_space() -> None:
    fixed_distributions = {
        "x": distributions.FloatDistribution(0, 1),
        "y": distributions.CategoricalDistribution(["bacon", "spam"]),
    }

    study = create_study()
    trial = study.ask(fixed_distributions=fixed_distributions)

    params = trial.params
    assert len(trial.params) == 2
    assert 0 <= params["x"] < 1
    assert params["y"] in ["bacon", "spam"]


# Deprecated distributions are internally converted to corresponding distributions.
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_ask_distribution_conversion() -> None:
    fixed_distributions = {
        "ud": distributions.UniformDistribution(low=0, high=10),
        "dud": distributions.DiscreteUniformDistribution(low=0, high=10, q=2),
        "lud": distributions.LogUniformDistribution(low=1, high=10),
        "id": distributions.IntUniformDistribution(low=0, high=10),
        "idd": distributions.IntUniformDistribution(low=0, high=10, step=2),
        "ild": distributions.IntLogUniformDistribution(low=1, high=10),
    }

    study = create_study()

    with pytest.warns(
        FutureWarning,
        match="See https://github.com/optuna/optuna/issues/2941",
    ) as record:
        trial = study.ask(fixed_distributions=fixed_distributions)
        assert len(record) == 6

    expected_distributions = {
        "ud": distributions.FloatDistribution(low=0, high=10, log=False, step=None),
        "dud": distributions.FloatDistribution(low=0, high=10, log=False, step=2),
        "lud": distributions.FloatDistribution(low=1, high=10, log=True, step=None),
        "id": distributions.IntDistribution(low=0, high=10, log=False, step=1),
        "idd": distributions.IntDistribution(low=0, high=10, log=False, step=2),
        "ild": distributions.IntDistribution(low=1, high=10, log=True, step=1),
    }

    assert trial.distributions == expected_distributions


# It confirms that ask doesn't convert non-deprecated distributions.
def test_ask_distribution_conversion_noop() -> None:
    fixed_distributions = {
        "ud": distributions.FloatDistribution(low=0, high=10, log=False, step=None),
        "dud": distributions.FloatDistribution(low=0, high=10, log=False, step=2),
        "lud": distributions.FloatDistribution(low=1, high=10, log=True, step=None),
        "id": distributions.IntDistribution(low=0, high=10, log=False, step=1),
        "idd": distributions.IntDistribution(low=0, high=10, log=False, step=2),
        "ild": distributions.IntDistribution(low=1, high=10, log=True, step=1),
        "cd": distributions.CategoricalDistribution(choices=["a", "b", "c"]),
    }

    study = create_study()

    trial = study.ask(fixed_distributions=fixed_distributions)

    # Check fixed_distributions doesn't change.
    assert trial.distributions == fixed_distributions


def test_tell() -> None:
    study = create_study()
    assert len(study.trials) == 0

    trial = study.ask()
    assert len(study.trials) == 1
    assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 0

    study.tell(trial, 1.0)
    assert len(study.trials) == 1
    assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 1

    study.tell(study.ask(), [1.0])
    assert len(study.trials) == 2
    assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 2

    # `trial` could be int.
    study.tell(study.ask().number, 1.0)
    assert len(study.trials) == 3
    assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 3

    # Inf is supported as values.
    study.tell(study.ask(), float("inf"))
    assert len(study.trials) == 4
    assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 4

    study.tell(study.ask(), state=TrialState.PRUNED)
    assert len(study.trials) == 5
    assert len(study.get_trials(states=(TrialState.PRUNED,))) == 1

    study.tell(study.ask(), state=TrialState.FAIL)
    assert len(study.trials) == 6
    assert len(study.get_trials(states=(TrialState.FAIL,))) == 1


def test_tell_pruned() -> None:
    study = create_study()

    study.tell(study.ask(), state=TrialState.PRUNED)
    assert study.trials[-1].value is None
    assert study.trials[-1].state == TrialState.PRUNED

    # Store the last intermediates as value.
    trial = study.ask()
    trial.report(2.0, step=1)
    study.tell(trial, state=TrialState.PRUNED)
    assert study.trials[-1].value == 2.0
    assert study.trials[-1].state == TrialState.PRUNED

    # Inf is also supported as a value.
    trial = study.ask()
    trial.report(float("inf"), step=1)
    study.tell(trial, state=TrialState.PRUNED)
    assert study.trials[-1].value == float("inf")
    assert study.trials[-1].state == TrialState.PRUNED

    # NaN is not supported as a value.
    trial = study.ask()
    trial.report(float("nan"), step=1)
    study.tell(trial, state=TrialState.PRUNED)
    assert study.trials[-1].value is None
    assert study.trials[-1].state == TrialState.PRUNED


def test_tell_automatically_fail() -> None:
    study = create_study()

    # Check invalid values, e.g. str cannot be cast to float.
    with pytest.warns(UserWarning):
        study.tell(study.ask(), "a")  # type: ignore
        assert len(study.trials) == 1
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    # Check invalid values, e.g. `None` that cannot be cast to float.
    with pytest.warns(UserWarning):
        study.tell(study.ask(), None)
        assert len(study.trials) == 2
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    # Check number of values.
    with pytest.warns(UserWarning):
        study.tell(study.ask(), [])
        assert len(study.trials) == 3
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    # Check wrong number of values, e.g. two values for single direction.
    with pytest.warns(UserWarning):
        study.tell(study.ask(), [1.0, 2.0])
        assert len(study.trials) == 4
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    # Both state and values are not specified.
    with pytest.warns(UserWarning):
        study.tell(study.ask())
        assert len(study.trials) == 5
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    # Nan is not supported.
    with pytest.warns(UserWarning):
        study.tell(study.ask(), float("nan"))
        assert len(study.trials) == 6
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None


def test_tell_multi_objective() -> None:
    study = create_study(directions=["minimize", "maximize"])
    study.tell(study.ask(), [1.0, 2.0])
    assert len(study.trials) == 1


def test_tell_multi_objective_automatically_fail() -> None:
    # Number of values doesn't match the length of directions.
    study = create_study(directions=["minimize", "maximize"])

    with pytest.warns(UserWarning):
        study.tell(study.ask(), [])
        assert len(study.trials) == 1
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    with pytest.warns(UserWarning):
        study.tell(study.ask(), [1.0])
        assert len(study.trials) == 2
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    with pytest.warns(UserWarning):
        study.tell(study.ask(), [1.0, 2.0, 3.0])
        assert len(study.trials) == 3
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    with pytest.warns(UserWarning):
        study.tell(study.ask(), [1.0, None])  # type: ignore
        assert len(study.trials) == 4
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    with pytest.warns(UserWarning):
        study.tell(study.ask(), [None, None])  # type: ignore
        assert len(study.trials) == 5
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    with pytest.warns(UserWarning):
        study.tell(study.ask(), 1.0)
        assert len(study.trials) == 6
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None


def test_tell_invalid() -> None:
    study = create_study()

    # Missing values for completions.
    with pytest.raises(ValueError):
        study.tell(study.ask(), state=TrialState.COMPLETE)

    # Invalid values for completions.
    with pytest.raises(ValueError):
        study.tell(study.ask(), "a", state=TrialState.COMPLETE)  # type: ignore

    with pytest.raises(ValueError):
        study.tell(study.ask(), None, state=TrialState.COMPLETE)

    with pytest.raises(ValueError):
        study.tell(study.ask(), [], state=TrialState.COMPLETE)

    with pytest.raises(ValueError):
        study.tell(study.ask(), [1.0, 2.0], state=TrialState.COMPLETE)

    with pytest.raises(ValueError):
        study.tell(study.ask(), float("nan"), state=TrialState.COMPLETE)

    # `state` must be None or finished state.
    with pytest.raises(ValueError):
        study.tell(study.ask(), state=TrialState.RUNNING)

    # `state` must be None or finished state.
    with pytest.raises(ValueError):
        study.tell(study.ask(), state=TrialState.WAITING)

    # `value` must be None for `TrialState.PRUNED`.
    with pytest.raises(ValueError):
        study.tell(study.ask(), values=1, state=TrialState.PRUNED)

    # `value` must be None for `TrialState.FAIL`.
    with pytest.raises(ValueError):
        study.tell(study.ask(), values=1, state=TrialState.FAIL)

    # Trial that has not been asked for cannot be told.
    with pytest.raises(ValueError):
        study.tell(study.ask().number + 1, 1.0)

    # Waiting trial cannot be told.
    with pytest.raises(ValueError):
        study.enqueue_trial({})
        study.tell(study.trials[-1].number, 1.0)

    # It must be Trial or int for trial.
    with pytest.raises(TypeError):
        study.tell("1", 1.0)  # type: ignore


def test_tell_duplicate_tell() -> None:
    study = create_study()

    trial = study.ask()
    study.tell(trial, 1.0)

    # Should not panic when passthrough is enabled.
    study.tell(trial, 1.0, skip_if_finished=True)

    with pytest.raises(ValueError):
        study.tell(trial, 1.0, skip_if_finished=False)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_enqueued_trial_datetime_start(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        def objective(trial: Trial) -> float:
            time.sleep(1)
            x = trial.suggest_int("x", -10, 10)
            return x

        study.enqueue_trial(params={"x": 1})
        assert study.trials[0].datetime_start is None

        study.optimize(objective, n_trials=1)
        assert study.trials[0].datetime_start is not None


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_study_summary_datetime_start_calculation(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:

        def objective(trial: Trial) -> float:
            x = trial.suggest_int("x", -10, 10)
            return x

        # StudySummary datetime_start tests.
        study = create_study(storage=storage)
        study.enqueue_trial(params={"x": 1})

        # Study summary with only enqueued trials should have null datetime_start.
        summaries = get_all_study_summaries(study._storage, include_best_trial=True)
        assert summaries[0].datetime_start is None

        # Study summary with completed trials should have nonnull datetime_start.
        study.optimize(objective, n_trials=1)
        study.enqueue_trial(params={"x": 1}, skip_if_exists=False)
        summaries = get_all_study_summaries(study._storage, include_best_trial=True)
        assert summaries[0].datetime_start is not None


def _process_tell(study: Study, trial: Trial | int, values: float) -> None:
    study.tell(trial, values)


def test_tell_from_another_process() -> None:
    pool = multiprocessing.Pool()

    with StorageSupplier("sqlite") as storage:
        # Create a study and ask for a new trial.
        study = create_study(storage=storage)
        trial0 = study.ask()

        # Test normal behaviour.
        pool.starmap(_process_tell, [(study, trial0, 1.2)])

        assert len(study.trials) == 1
        assert study.best_trial.state == TrialState.COMPLETE
        assert study.best_value == 1.2

        # Test study.tell using trial number.
        trial = study.ask()
        pool.starmap(_process_tell, [(study, trial.number, 1.5)])

        assert len(study.trials) == 2
        assert study.best_trial.state == TrialState.COMPLETE
        assert study.best_value == 1.2

        # Should fail because the trial0 is already finished.
        with pytest.raises(ValueError):
            pool.starmap(_process_tell, [(study, trial0, 1.2)])


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_pop_waiting_trial_thread_safe(storage_mode: str) -> None:
    if storage_mode in ("sqlite", "cached_sqlite", "grpc_rdb"):
        pytest.skip("study._pop_waiting_trial is not thread-safe on SQLite3")

    num_enqueued = 10
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        for i in range(num_enqueued):
            study.enqueue_trial({"i": i})

        trial_id_set = set()
        with ThreadPoolExecutor(10) as pool:
            futures = []
            for i in range(num_enqueued):
                future = pool.submit(study._pop_waiting_trial_id)
                futures.append(future)

            for future in as_completed(futures):
                trial_id_set.add(future.result())
        assert len(trial_id_set) == num_enqueued


def test_pop_waiting_trial_id_race_condition() -> None:
    study = create_study()
    study.add_trial(create_trial(state=TrialState.COMPLETE, value=0))
    study.add_trial(create_trial(state=TrialState.RUNNING))
    study.add_trial(create_trial(state=TrialState.WAITING))
    trials = study.get_trials(deepcopy=False)

    # Return all trials as waiting to emulate the race condition.
    study._storage.get_all_trials = MagicMock(return_value=trials)  # type: ignore[method-assign]

    trial_id = study._pop_waiting_trial_id()
    assert trial_id == study.trials[2]._trial_id


def test_set_metric_names() -> None:
    metric_names = ["v0", "v1"]
    study = create_study(directions=["minimize", "minimize"])
    study.set_metric_names(metric_names)

    got_metric_names = study._storage.get_study_system_attrs(study._study_id).get(
        _SYSTEM_ATTR_METRIC_NAMES
    )
    assert got_metric_names is not None
    assert metric_names == got_metric_names


def test_set_metric_names_experimental_warning() -> None:
    study = create_study()
    with pytest.warns(ExperimentalWarning):
        study.set_metric_names(["v0"])


def test_set_invalid_metric_names() -> None:
    metric_names = ["v0", "v1", "v2"]
    study = create_study(directions=["minimize", "minimize"])
    with pytest.raises(ValueError):
        study.set_metric_names(metric_names)


def test_get_metric_names() -> None:
    study = create_study()
    assert study.metric_names is None
    study.set_metric_names(["v0"])
    assert study.metric_names == ["v0"]
    study.set_metric_names(["v1"])
    assert study.metric_names == ["v1"]
