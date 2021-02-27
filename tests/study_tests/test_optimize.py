import itertools
import multiprocessing
import threading
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from unittest.mock import Mock  # NOQA
from unittest.mock import patch

import pytest

import optuna
from optuna import _optimize
from optuna import create_study
from optuna import Trial
from optuna import TrialPruned
from optuna.exceptions import TrialPruned as TrialPruned_in_exceptions
from optuna.structs import TrialPruned as TrialPruned_in_structs
from optuna.study import StudyDirection
from optuna.testing.storage import STORAGE_MODES
from optuna.testing.storage import StorageSupplier
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def func(trial: Trial, x_max: float = 1.0) -> float:

    x = trial.suggest_float("x", -x_max, x_max)
    y = trial.suggest_float("y", 20, 30, log=True)
    z = trial.suggest_categorical("z", (-1.0, 1.0))
    assert isinstance(z, float)
    return (x - 2) ** 2 + (y - 25) ** 2 + z


class Func(object):
    def __init__(self, sleep_sec: Optional[float] = None) -> None:

        self.n_calls = 0
        self.sleep_sec = sleep_sec
        self.lock = threading.Lock()
        self.x_max = 10.0

    def __call__(self, trial: Trial) -> float:

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


def check_params(params: Dict[str, Any]) -> None:
    assert sorted(params.keys()) == ["x", "y", "z"]


def check_value(value: Optional[float]) -> None:

    assert isinstance(value, float)
    assert -1.0 <= value <= 12.0 ** 2 + 5.0 ** 2 + 1.0


def check_frozen_trial(frozen_trial: FrozenTrial) -> None:

    if frozen_trial.state == TrialState.COMPLETE:
        check_params(frozen_trial.params)
        check_value(frozen_trial.value)


def check_study(study: optuna.Study) -> None:

    for trial in study.trials:
        check_frozen_trial(trial)

    assert not study._is_multi_objective()

    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
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


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        # Test trial with acceptable exception.
        def func_value_error(_: Trial) -> float:

            raise ValueError

        trial = _optimize._run_trial(study, func_value_error, catch=(ValueError,))
        frozen_trial = study._storage.get_trial(trial._trial_id)

        assert frozen_trial.state == TrialState.FAIL

        # Test trial with unacceptable exception.
        with pytest.raises(ValueError):
            _optimize._run_trial(study, func_value_error, catch=(ArithmeticError,))

        # Test trial with invalid objective value: None
        def func_none(_: Trial) -> float:

            return None  # type: ignore

        trial = _optimize._run_trial(study, func_none, catch=(Exception,))
        frozen_trial = study._storage.get_trial(trial._trial_id)

        assert frozen_trial.state == TrialState.FAIL

        # Test trial with invalid objective value: nan
        def func_nan(_: Trial) -> float:

            return float("nan")

        trial = _optimize._run_trial(study, func_nan, catch=(Exception,))
        frozen_trial = study._storage.get_trial(trial._trial_id)

        assert frozen_trial.state == TrialState.FAIL


# TODO(Yanase): Remove this test function after removing `optuna.structs.TrialPruned`.
@pytest.mark.parametrize(
    "trial_pruned_class",
    [TrialPruned, TrialPruned_in_exceptions, TrialPruned_in_structs],
)
@pytest.mark.parametrize("report_value", [None, 1.2])
def test_run_trial_with_trial_pruned(
    trial_pruned_class: Callable[[], TrialPruned], report_value: Optional[float]
) -> None:

    study = create_study()

    def func_with_trial_pruned(trial: Trial) -> float:

        if report_value is not None:
            trial.report(report_value, 1)

        raise trial_pruned_class()

    trial = _optimize._run_trial(study, func_with_trial_pruned, catch=())
    frozen_trial = study._storage.get_trial(trial._trial_id)
    assert frozen_trial.value == report_value
    assert frozen_trial.state == TrialState.PRUNED


def test_trivial_in_memory_new() -> None:

    study = create_study()
    study.optimize(func, n_trials=10)
    check_study(study)


def test_trivial_in_memory_resume() -> None:

    study = create_study()
    study.optimize(func, n_trials=10)
    study.optimize(func, n_trials=10)
    check_study(study)


def test_trivial_rdb_resume_study() -> None:

    study = create_study("sqlite:///:memory:")
    study.optimize(func, n_trials=10)
    check_study(study)


def test_direction() -> None:

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


@pytest.mark.parametrize(
    "n_trials, n_jobs, storage_mode",
    itertools.product((0, 1, 20), (1, 2, -1), STORAGE_MODES),  # n_trials  # n_jobs  # storage_mode
)
def test_optimize_parallel(n_trials: int, n_jobs: int, storage_mode: str) -> None:

    f = Func()

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(f, n_trials=n_trials, n_jobs=n_jobs)
        assert f.n_calls == len(study.trials) == n_trials
        check_study(study)


@pytest.mark.parametrize(
    "n_trials, n_jobs, storage_mode",
    itertools.product(
        (0, 1, 20, None), (1, 2, -1), STORAGE_MODES  # n_trials  # n_jobs  # storage_mode
    ),
)
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
def test_catch(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        def func_value_error(_: Trial) -> float:

            raise ValueError

        # Test default exceptions.
        with pytest.raises(ValueError):
            study.optimize(func_value_error, n_trials=20)
        assert len(study.trials) == 1
        assert all(trial.state == TrialState.FAIL for trial in study.trials)

        # Test acceptable exception.
        study.optimize(func_value_error, n_trials=20, catch=(ValueError,))
        assert len(study.trials) == 21
        assert all(trial.state == TrialState.FAIL for trial in study.trials)

        # Test trial with unacceptable exception.
        with pytest.raises(ValueError):
            study.optimize(func_value_error, n_trials=20, catch=(ArithmeticError,))
        assert len(study.trials) == 22
        assert all(trial.state == TrialState.FAIL for trial in study.trials)


@pytest.mark.parametrize("catch", [[], [Exception], None, 1])
def test_catch_invalid_type(catch: Any) -> None:

    study = create_study()

    def func_value_error(_: Trial) -> float:

        raise ValueError

    with pytest.raises(TypeError):
        study.optimize(func_value_error, n_trials=20, catch=catch)


@pytest.mark.parametrize(
    "n_jobs, storage_mode", itertools.product((2, -1), STORAGE_MODES)  # n_jobs  # storage_mode
)
def test_reseeding(n_jobs: int, storage_mode: str) -> None:

    f = Func()

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        sampler = study.sampler
        with patch.object(sampler, "reseed_rng", wraps=sampler.reseed_rng) as mock_object:
            study.optimize(f, n_trials=1, n_jobs=2)
            assert mock_object.call_count == 1


def test_stop_outside_optimize() -> None:
    # Test stopping outside the optimization: it should raise `RuntimeError`.
    study = create_study()
    with pytest.raises(RuntimeError):
        study.stop()

    # Test calling `optimize` after the `RuntimeError` is caught.
    study.optimize(lambda _: 1.0, n_trials=1)


@patch("optuna._optimize.gc.collect")
def test_gc(collect_mock: Mock) -> None:

    study = create_study()
    study.optimize(func, n_trials=10, gc_after_trial=True)
    check_study(study)
    assert collect_mock.call_count == 10


@patch("optuna._optimize.gc.collect")
def test_no_gc(collect_mock: Mock) -> None:

    study = create_study()
    study.optimize(func, n_trials=10, gc_after_trial=False)
    check_study(study)
    assert collect_mock.call_count == 0


@pytest.mark.parametrize("n_objectives", [2, 3])
def test_multi_objectives(n_objectives: int) -> None:
    directions = ["minimize" for _ in range(n_objectives)]
    study = create_study(directions=directions)

    def objective(trial: Trial) -> List[float]:
        return [trial.suggest_uniform("v{}".format(i), 0, 5) for i in range(n_objectives)]

    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 10

    for trial in study.trials:
        assert trial.values
        assert len(trial.values) == n_objectives


@pytest.mark.parametrize("n_objectives", [1, 2, 3])
def test_optimize(n_objectives: int) -> None:
    directions = ["minimize" for _ in range(n_objectives)]
    study = optuna.multi_objective.create_study(directions)

    def objective(trial: optuna.multi_objective.trial.MultiObjectiveTrial) -> List[float]:
        return [trial.suggest_uniform("v{}".format(i), 0, 5) for i in range(n_objectives)]

    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 10

    for trial in study.trials:
        assert len(trial.values) == n_objectives
