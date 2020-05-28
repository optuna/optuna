import copy
import itertools
import multiprocessing
import pickle
import threading
import time
from unittest.mock import Mock  # NOQA
from unittest.mock import patch
import uuid
import warnings

import joblib
import pandas as pd
import pytest

import optuna
from optuna.testing.storage import StorageSupplier
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Any  # NOQA
    from typing import Callable  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA
    from typing import Tuple  # NOQA

    from _pytest.recwarn import WarningsRecorder  # NOQA

    CallbackFuncType = Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]

# TODO(ytsmiling) Add tests for multi-worker settings.
STORAGE_MODES = [
    "inmemory",
    "sqlite",
    "redis",
]


def func(trial, x_max=1.0):
    # type: (optuna.trial.Trial, float) -> float

    x = trial.suggest_uniform("x", -x_max, x_max)
    y = trial.suggest_loguniform("y", 20, 30)
    z = trial.suggest_categorical("z", (-1.0, 1.0))
    assert isinstance(z, float)
    return (x - 2) ** 2 + (y - 25) ** 2 + z


class Func(object):
    def __init__(self, sleep_sec=None):
        # type: (Optional[float]) -> None

        self.n_calls = 0
        self.sleep_sec = sleep_sec
        self.lock = threading.Lock()
        self.x_max = 10.0

    def __call__(self, trial):
        # type: (optuna.trial.Trial) -> float

        with self.lock:
            self.n_calls += 1
            x_max = self.x_max
            self.x_max *= 0.9

        # Sleep for testing parallelism
        if self.sleep_sec is not None:
            time.sleep(self.sleep_sec)

        value = func(trial, x_max)
        check_params(trial.params)
        return value


def check_params(params):
    # type: (Dict[str, Any]) -> None

    assert sorted(params.keys()) == ["x", "y", "z"]


def check_value(value):
    # type: (Optional[float]) -> None

    assert isinstance(value, float)
    assert -1.0 <= value <= 12.0 ** 2 + 5.0 ** 2 + 1.0


def check_frozen_trial(frozen_trial):
    # type: (optuna.trial.FrozenTrial) -> None

    if frozen_trial.state == optuna.trial.TrialState.COMPLETE:
        check_params(frozen_trial.params)
        check_value(frozen_trial.value)


def check_study(study):
    # type: (optuna.Study) -> None

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


def test_optimize_trivial_in_memory_new():
    # type: () -> None

    study = optuna.create_study()
    study.optimize(func, n_trials=10)
    check_study(study)


def test_optimize_trivial_in_memory_resume():
    # type: () -> None

    study = optuna.create_study()
    study.optimize(func, n_trials=10)
    study.optimize(func, n_trials=10)
    check_study(study)


def test_optimize_trivial_rdb_resume_study():
    # type: () -> None

    study = optuna.create_study("sqlite:///:memory:")
    study.optimize(func, n_trials=10)
    check_study(study)


def test_optimize_with_direction():
    # type: () -> None

    study = optuna.create_study(direction="minimize")
    study.optimize(func, n_trials=10)
    assert study.direction == optuna.study.StudyDirection.MINIMIZE
    check_study(study)

    study = optuna.create_study(direction="maximize")
    study.optimize(func, n_trials=10)
    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    check_study(study)

    with pytest.raises(ValueError):
        optuna.create_study(direction="test")


@pytest.mark.parametrize(
    "n_trials, n_jobs, storage_mode",
    itertools.product(
        (0, 1, 20), (1, 2, -1), STORAGE_MODES,  # n_trials  # n_jobs  # storage_mode
    ),
)
def test_optimize_parallel(n_trials, n_jobs, storage_mode):
    # type: (int, int, str)-> None

    f = Func()

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        study.optimize(f, n_trials=n_trials, n_jobs=n_jobs)
        assert f.n_calls == len(study.trials) == n_trials
        check_study(study)


@pytest.mark.parametrize(
    "n_trials, n_jobs, storage_mode",
    itertools.product(
        (0, 1, 20, None), (1, 2, -1), STORAGE_MODES,  # n_trials  # n_jobs  # storage_mode
    ),
)
def test_optimize_parallel_timeout(n_trials, n_jobs, storage_mode):
    # type: (int, int, str) -> None

    sleep_sec = 0.1
    timeout_sec = 1.0
    f = Func(sleep_sec=sleep_sec)

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
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
def test_optimize_with_catch(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)

        def func_value_error(_):
            # type: (optuna.trial.Trial) -> float

            raise ValueError

        # Test default exceptions.
        with pytest.raises(ValueError):
            study.optimize(func_value_error, n_trials=20)
        assert len(study.trials) == 1
        assert all(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials)

        # Test acceptable exception.
        study.optimize(func_value_error, n_trials=20, catch=(ValueError,))
        assert len(study.trials) == 21
        assert all(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials)

        # Test trial with unacceptable exception.
        with pytest.raises(ValueError):
            study.optimize(func_value_error, n_trials=20, catch=(ArithmeticError,))
        assert len(study.trials) == 22
        assert all(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials)


@pytest.mark.parametrize("catch", [[], [Exception], None, 1])
def test_optimize_with_catch_invalid_type(catch):
    # type: (Any) -> None

    study = optuna.create_study()

    def func_value_error(_):
        # type: (optuna.trial.Trial) -> float

        raise ValueError

    with pytest.raises(TypeError):
        study.optimize(func_value_error, n_trials=20, catch=catch)


def test_optimize_parallel_storage_warning(recwarn):
    # type: (WarningsRecorder) -> None

    study = optuna.create_study()

    # Default joblib backend is threading and no warnings will be captured.
    study.optimize(lambda t: t.suggest_uniform("x", 0, 1), n_trials=20, n_jobs=2)
    assert len(recwarn) == 0

    with pytest.warns(UserWarning):
        with joblib.parallel_backend("loky"):
            study.optimize(lambda t: t.suggest_uniform("x", 0, 1), n_trials=20, n_jobs=2)


@pytest.mark.parametrize(
    "n_jobs, storage_mode", itertools.product((2, -1), STORAGE_MODES,),  # n_jobs  # storage_mode
)
def test_optimize_with_reseeding(n_jobs, storage_mode):
    # type: (int, str)-> None

    f = Func()

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        sampler = study.sampler
        with patch.object(sampler, "reseed_rng", wraps=sampler.reseed_rng) as mock_object:
            study.optimize(f, n_trials=1, n_jobs=2)
            assert mock_object.call_count == 1


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_study_set_and_get_user_attrs(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)

        study.set_user_attr("dataset", "MNIST")
        assert study.user_attrs["dataset"] == "MNIST"


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_study_set_and_get_system_attrs(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)

        study.set_system_attr("system_message", "test")
        assert study.system_attrs["system_message"] == "test"


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_trial_set_and_get_user_attrs(storage_mode):
    # type: (str) -> None

    def f(trial):
        # type: (optuna.trial.Trial) -> float

        trial.set_user_attr("train_accuracy", 1)
        assert trial.user_attrs["train_accuracy"] == 1
        return 0.0

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        study.optimize(f, n_trials=1)
        frozen_trial = study.trials[0]
        assert frozen_trial.user_attrs["train_accuracy"] == 1


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_trial_set_and_get_system_attrs(storage_mode):
    # type: (str) -> None

    def f(trial):
        # type: (optuna.trial.Trial) -> float

        trial.set_system_attr("system_message", "test")
        assert trial.system_attrs["system_message"] == "test"
        return 0.0

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        study.optimize(f, n_trials=1)
        frozen_trial = study.trials[0]
        assert frozen_trial.system_attrs["system_message"] == "test"


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_all_study_summaries(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        study.optimize(Func(), n_trials=5)

        summaries = optuna.get_all_study_summaries(study._storage)
        summary = [s for s in summaries if s._study_id == study._study_id][0]

        assert summary.study_name == study.study_name
        assert summary.n_trials == 5


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_all_study_summaries_with_no_trials(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)

        summaries = optuna.get_all_study_summaries(study._storage)
        summary = [s for s in summaries if s._study_id == study._study_id][0]

        assert summary.study_name == study.study_name
        assert summary.n_trials == 0
        assert summary.datetime_start is None


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)

        # Test trial without exception.
        study._run_trial(func, catch=(Exception,), gc_after_trial=True)
        check_study(study)

        # Test trial with acceptable exception.
        def func_value_error(_):
            # type: (optuna.trial.Trial) -> float

            raise ValueError

        trial = study._run_trial(func_value_error, catch=(ValueError,), gc_after_trial=True)
        frozen_trial = study._storage.get_trial(trial._trial_id)

        expected_message = (
            "Setting status of trial#1 as TrialState.FAIL because of the "
            "following error: ValueError()"
        )
        assert frozen_trial.state == optuna.trial.TrialState.FAIL
        assert frozen_trial.system_attrs["fail_reason"] == expected_message

        # Test trial with unacceptable exception.
        with pytest.raises(ValueError):
            study._run_trial(func_value_error, catch=(ArithmeticError,), gc_after_trial=True)

        # Test trial with invalid objective value: None
        def func_none(_):
            # type: (optuna.trial.Trial) -> float

            return None  # type: ignore

        trial = study._run_trial(func_none, catch=(Exception,), gc_after_trial=True)
        frozen_trial = study._storage.get_trial(trial._trial_id)

        expected_message = (
            "Setting status of trial#3 as TrialState.FAIL because the returned "
            "value from the objective function cannot be casted to float. "
            "Returned value is: None"
        )
        assert frozen_trial.state == optuna.trial.TrialState.FAIL
        assert frozen_trial.system_attrs["fail_reason"] == expected_message

        # Test trial with invalid objective value: nan
        def func_nan(_):
            # type: (optuna.trial.Trial) -> float

            return float("nan")

        trial = study._run_trial(func_nan, catch=(Exception,), gc_after_trial=True)
        frozen_trial = study._storage.get_trial(trial._trial_id)

        expected_message = (
            "Setting status of trial#4 as TrialState.FAIL because the objective "
            "function returned nan."
        )
        assert frozen_trial.state == optuna.trial.TrialState.FAIL
        assert frozen_trial.system_attrs["fail_reason"] == expected_message


# TODO(Yanase): Remove this test function after removing `optuna.structs.TrialPruned`.
@pytest.mark.parametrize(
    "trial_pruned_class",
    [optuna.TrialPruned, optuna.exceptions.TrialPruned, optuna.structs.TrialPruned],
)
@pytest.mark.parametrize("report_value", [None, 1.2])
def test_run_trial_with_trial_pruned(trial_pruned_class, report_value):
    # type: (Callable[[], optuna.exceptions.TrialPruned], Optional[float]) -> None

    study = optuna.create_study()

    def func_with_trial_pruned(trial):
        # type: (optuna.trial.Trial) -> float

        if report_value is not None:
            trial.report(report_value, 1)

        raise trial_pruned_class()

    trial = study._run_trial(func_with_trial_pruned, catch=(), gc_after_trial=True)
    frozen_trial = study._storage.get_trial(trial._trial_id)
    assert frozen_trial.value == report_value
    assert frozen_trial.state == optuna.trial.TrialState.PRUNED


def test_study_pickle():
    # type: () -> None

    study_1 = optuna.create_study()
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


def test_study_trials_dataframe_with_no_trials():
    # type: () -> None

    study_with_no_trials = optuna.create_study()
    trials_df = study_with_no_trials.trials_dataframe()
    assert trials_df.empty


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
@pytest.mark.parametrize(
    "attrs",
    [
        (
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "params",
            "user_attrs",
            "system_attrs",
            "state",
        ),
        (
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "system_attrs",
            "state",
            "intermediate_values",
            "_trial_id",
            "distributions",
        ),
    ],
)
@pytest.mark.parametrize("multi_index", [True, False])
def test_trials_dataframe(storage_mode, attrs, multi_index):
    # type: (str, Tuple[str, ...], bool) -> None

    def f(trial):
        # type: (optuna.trial.Trial) -> float

        x = trial.suggest_int("x", 1, 1)
        y = trial.suggest_categorical("y", (2.5,))
        assert isinstance(y, float)
        trial.set_user_attr("train_loss", 3)
        trial.set_system_attr("foo", "bar")
        value = x + y  # 3.5

        # Test reported intermediate values, although it in practice is not "intermediate".
        trial.report(value, step=0)

        return value

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        study.optimize(f, n_trials=3)
        df = study.trials_dataframe(attrs=attrs, multi_index=multi_index)
        # Change index to access rows via trial number.
        if multi_index:
            df.set_index(("number", ""), inplace=True, drop=False)
        else:
            df.set_index("number", inplace=True, drop=False)
        assert len(df) == 3

        # Number columns are as follows (total of 13):
        #   non-nested: 6 (number, value, state, datetime_start, datetime_complete, duration)
        #   params: 2
        #   distributions: 2
        #   user_attrs: 1
        #   system_attrs: 1
        #   intermediate_values: 1
        expected_n_columns = len(attrs)
        if "params" in attrs:
            expected_n_columns += 1
        if "distributions" in attrs:
            expected_n_columns += 1
        assert len(df.columns) == expected_n_columns

        for i in range(3):
            assert df.number[i] == i
            assert df.state[i] == "COMPLETE"
            assert df.value[i] == 3.5
            assert isinstance(df.datetime_start[i], pd.Timestamp)
            assert isinstance(df.datetime_complete[i], pd.Timestamp)

            if multi_index:
                if "distributions" in attrs:
                    assert ("distributions", "x") in df.columns
                    assert ("distributions", "y") in df.columns
                if "_trial_id" in attrs:
                    assert ("trial_id", "") in df.columns  # trial_id depends on other tests.
                if "duration" in attrs:
                    assert ("duration", "") in df.columns

                assert df.params.x[i] == 1
                assert df.params.y[i] == 2.5
                assert df.user_attrs.train_loss[i] == 3
                assert df.system_attrs.foo[i] == "bar"
            else:
                if "distributions" in attrs:
                    assert "distributions_x" in df.columns
                    assert "distributions_y" in df.columns
                if "_trial_id" in attrs:
                    assert "trial_id" in df.columns  # trial_id depends on other tests.
                if "duration" in attrs:
                    assert "duration" in df.columns

                assert df.params_x[i] == 1
                assert df.params_y[i] == 2.5
                assert df.user_attrs_train_loss[i] == 3
                assert df.system_attrs_foo[i] == "bar"


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_trials_dataframe_with_failure(storage_mode):
    # type: (str) -> None

    def f(trial):
        # type: (optuna.trial.Trial) -> float

        x = trial.suggest_int("x", 1, 1)
        y = trial.suggest_categorical("y", (2.5,))
        trial.set_user_attr("train_loss", 3)
        raise ValueError()
        return x + y  # 3.5

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        study.optimize(f, n_trials=3, catch=(ValueError,))
        df = study.trials_dataframe()
        # Change index to access rows via trial number.
        df.set_index("number", inplace=True, drop=False)
        assert len(df) == 3
        # non-nested: 6, params: 2, user_attrs: 1 system_attrs: 1
        assert len(df.columns) == 10
        for i in range(3):
            assert df.number[i] == i
            assert df.state[i] == "FAIL"
            assert df.value[i] is None
            assert isinstance(df.datetime_start[i], pd.Timestamp)
            assert isinstance(df.datetime_complete[i], pd.Timestamp)
            assert isinstance(df.duration[i], pd.Timedelta)
            assert df.params_x[i] == 1
            assert df.params_y[i] == 2.5
            assert df.user_attrs_train_loss[i] == 3
            assert "system_attrs_fail_reason" in df.columns


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_create_study(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:
        # Test creating a new study.
        study = optuna.create_study(storage=storage, load_if_exists=False)

        # Test `load_if_exists=True` with existing study.
        optuna.create_study(study_name=study.study_name, storage=storage, load_if_exists=True)

        with pytest.raises(optuna.exceptions.DuplicatedStudyError):
            optuna.create_study(study_name=study.study_name, storage=storage, load_if_exists=False)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_load_study(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:
        if storage is None:
            # `InMemoryStorage` can not be used with `load_study` function.
            return

        study_name = str(uuid.uuid4())

        with pytest.raises(KeyError):
            # Test loading an unexisting study.
            optuna.study.load_study(study_name=study_name, storage=storage)

        # Create a new study.
        created_study = optuna.study.create_study(study_name=study_name, storage=storage)

        # Test loading an existing study.
        loaded_study = optuna.study.load_study(study_name=study_name, storage=storage)
        assert created_study._study_id == loaded_study._study_id


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_delete_study(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:
        # Get storage object because delete_study does not accept None.
        storage = optuna.storages.get_storage(storage=storage)
        assert storage is not None

        # Test deleting a non-existing study.
        with pytest.raises(KeyError):
            optuna.delete_study("invalid-study-name", storage)

        # Test deleting an existing study.
        study = optuna.create_study(storage=storage, load_if_exists=False)
        optuna.delete_study(study.study_name, storage)

        # Test failed to delete the study which is already deleted.
        with pytest.raises(KeyError):
            optuna.delete_study(study.study_name, storage)


def test_nested_optimization():
    # type: () -> None

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        with pytest.raises(RuntimeError):
            trial.study.optimize(lambda _: 0.0, n_trials=1)

        return 1.0

    study = optuna.create_study()
    study.optimize(objective, n_trials=10, catch=())


def test_stop_in_objective() -> None:
    def objective(trial: optuna.trial.Trial, threshold_number: int) -> float:
        if trial.number >= threshold_number:
            trial.study.stop()

        return trial.number

    # Test stopping the optimization: it should stop once the trial number reaches 4.
    study = optuna.create_study()
    study.optimize(lambda x: objective(x, 4), n_trials=10)
    assert len(study.trials) == 5

    # Test calling `optimize` again: it should stop once the trial number reaches 11.
    study.optimize(lambda x: objective(x, 11), n_trials=10)
    assert len(study.trials) == 12


def test_stop_in_callback() -> None:
    def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.number >= 4:
            study.stop()

    # Test stopping the optimization inside a callback.
    study = optuna.create_study()
    study.optimize(lambda _: 1.0, n_trials=10, callbacks=[callback])
    assert len(study.trials) == 5


def test_stop_n_jobs() -> None:
    def callback(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.number >= 4:
            study.stop()

    study = optuna.create_study()
    study.optimize(lambda _: 1.0, n_trials=None, callbacks=[callback], n_jobs=2)
    assert 5 <= len(study.trials) <= 6


def test_stop_outside_optimize() -> None:
    # Test stopping outside the optimization: it should raise `RuntimeError`.
    study = optuna.create_study()
    with pytest.raises(RuntimeError):
        study.stop()

    # Test calling `optimize` after the `RuntimeError` is caught.
    study.optimize(lambda _: 1.0, n_trials=1)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_append_trial(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        assert len(study.trials) == 0

        trial_id = study._append_trial(value=0.8)
        assert study.trials[0]._trial_id == trial_id
        assert len(study.trials) == 1
        assert study.best_value == 0.8


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_enqueue_trial_properly_sets_param_values(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        assert len(study.trials) == 0

        study.enqueue_trial(params={"x": -5, "y": 5})
        study.enqueue_trial(params={"x": -1, "y": 0})

        def objective(trial):
            # type: (optuna.trial.Trial) -> float

            x = trial.suggest_int("x", -10, 10)
            y = trial.suggest_int("y", -10, 10)
            return x ** 2 + y ** 2

        study.optimize(objective, n_trials=2)
        t0 = study.trials[0]
        assert t0.params["x"] == -5
        assert t0.params["y"] == 5

        t1 = study.trials[1]
        assert t1.params["x"] == -1
        assert t1.params["y"] == 0


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_enqueue_trial_with_unfixed_parameters(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:
        study = optuna.create_study(storage=storage)
        assert len(study.trials) == 0

        study.enqueue_trial(params={"x": -5})

        def objective(trial):
            # type: (optuna.trial.Trial) -> float

            x = trial.suggest_int("x", -10, 10)
            y = trial.suggest_int("y", -10, 10)
            return x ** 2 + y ** 2

        study.optimize(objective, n_trials=1)
        t = study.trials[0]
        assert t.params["x"] == -5
        assert -10 <= t.params["y"] <= 10


def test_storage_property():
    # type: () -> None

    study = optuna.create_study()
    assert study.storage == study._storage


@patch("optuna.study.gc.collect")
def test_optimize_with_gc(collect_mock):
    # type: (Mock) -> None

    study = optuna.create_study()
    study.optimize(func, n_trials=10, gc_after_trial=True)
    check_study(study)
    assert collect_mock.call_count == 10


@patch("optuna.study.gc.collect")
def test_optimize_without_gc(collect_mock):
    # type: (Mock) -> None

    study = optuna.create_study()
    study.optimize(func, n_trials=10, gc_after_trial=False)
    check_study(study)
    assert collect_mock.call_count == 0


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_callbacks(n_jobs):
    # type: (int) -> None

    lock = threading.Lock()

    def with_lock(f):
        # type: (CallbackFuncType) -> CallbackFuncType

        def callback(study, trial):
            # type: (optuna.study.Study, optuna.trial.FrozenTrial) -> None

            with lock:
                f(study, trial)

        return callback

    study = optuna.create_study()

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        return trial.suggest_int("x", 1, 1)

    # Empty callback list.
    study.optimize(objective, callbacks=[], n_trials=10, n_jobs=n_jobs)

    # A callback.
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
    assert states == [optuna.trial.TrialState.FAIL] * 10

    # If a trial is failed with an exception and the exception isn't caught by the study,
    # callbacks aren't invoked.
    states = []
    callbacks = [with_lock(lambda study, trial: states.append(trial.state))]
    with pytest.raises(ZeroDivisionError):
        study.optimize(lambda t: 1 / 0, callbacks=callbacks, n_trials=10, n_jobs=n_jobs, catch=())
    assert states == []


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_get_trials(storage_mode):
    # type: (str) -> None

    with StorageSupplier(storage_mode) as storage:
        storage = optuna.storages.get_storage(storage=storage)

        study = optuna.create_study(storage=storage)
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


def test_study_id():
    # type: () -> None

    study = optuna.create_study()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        assert study.study_id == study._study_id

    with pytest.warns(DeprecationWarning):
        study.study_id


def test_study_summary_study_id():
    # type: () -> None

    study = optuna.create_study()
    summaries = study._storage.get_all_study_summaries()
    assert len(summaries) == 1

    summary = summaries[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        assert summary.study_id == summary._study_id

    with pytest.warns(DeprecationWarning):
        summary.study_id


def test_study_summary_eq_ne():
    # type: () -> None

    storage = optuna.storages.RDBStorage("sqlite:///:memory:")

    optuna.create_study(storage=storage)
    study = optuna.create_study(storage=storage)

    summaries = study._storage.get_all_study_summaries()
    assert len(summaries) == 2

    assert summaries[0] == copy.deepcopy(summaries[0])
    assert not summaries[0] != copy.deepcopy(summaries[0])

    assert not summaries[0] == summaries[1]
    assert summaries[0] != summaries[1]

    assert not summaries[0] == 1
    assert summaries[0] != 1


def test_study_summary_lt_le():
    # type: () -> None

    storage = optuna.storages.RDBStorage("sqlite:///:memory:")

    optuna.create_study(storage=storage)
    study = optuna.create_study(storage=storage)

    summaries = study._storage.get_all_study_summaries()
    assert len(summaries) == 2

    summary_0 = summaries[0]
    summary_1 = summaries[1]

    assert summary_0 < summary_1
    assert not summary_1 < summary_0

    with pytest.raises(TypeError):
        summary_0 < 1

    assert summary_0 <= summary_0
    assert not summary_1 <= summary_0

    with pytest.raises(TypeError):
        summary_0 <= 1

    # A list of StudySummaries is sortable.
    summaries.reverse()
    summaries.sort()
    assert summaries[0] == summary_0
    assert summaries[1] == summary_1
