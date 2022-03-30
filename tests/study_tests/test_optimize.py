import itertools
import multiprocessing
import threading
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from unittest import mock
from unittest.mock import Mock
from unittest.mock import patch

import _pytest.capture
import pytest

from optuna import create_study
from optuna import Study
from optuna import Trial
from optuna import TrialPruned
from optuna.study import _optimize
from optuna.study import StudyDirection
from optuna.study._tell import _tell_with_warning
from optuna.testing.storage import STORAGE_MODES
from optuna.testing.storage import StorageSupplier
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


CallbackFuncType = Callable[[Study, FrozenTrial], None]


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
    assert -1.0 <= value <= 12.0**2 + 5.0**2 + 1.0


def check_frozen_trial(frozen_trial: FrozenTrial) -> None:

    if frozen_trial.state == TrialState.COMPLETE:
        check_params(frozen_trial.params)
        check_value(frozen_trial.value)


def check_study(study: Study) -> None:

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
def test_optimize_with_catch(storage_mode: str) -> None:

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
def test_optimize_with_catch_invalid_type(catch: Any) -> None:

    study = create_study()

    def func_value_error(_: Trial) -> float:

        raise ValueError

    with pytest.raises(TypeError):
        study.optimize(func_value_error, n_trials=20, catch=catch)


@pytest.mark.parametrize(
    "n_jobs, storage_mode", itertools.product((2, -1), STORAGE_MODES)  # n_jobs  # storage_mode
)
def test_optimize_with_reseeding(n_jobs: int, storage_mode: str) -> None:

    f = Func()

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        sampler = study.sampler
        with patch.object(sampler, "reseed_rng", wraps=sampler.reseed_rng) as mock_object:
            study.optimize(f, n_trials=1, n_jobs=2)
            assert mock_object.call_count == 1


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
    assert "10/10" in err
    assert "100%" in err


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_optimize_without_progbar(n_jobs: int, capsys: _pytest.capture.CaptureFixture) -> None:

    study = create_study()
    study.optimize(lambda _: 1.0, n_trials=10, n_jobs=n_jobs)
    _, err = capsys.readouterr()

    assert "10/10" not in err
    assert "100%" not in err


def test_optimize_with_progbar_timeout(capsys: _pytest.capture.CaptureFixture) -> None:

    study = create_study()
    study.optimize(lambda _: 1.0, timeout=2.0, show_progress_bar=True)
    _, err = capsys.readouterr()

    assert "00:02/00:02" in err
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
    def _objective(trial: Trial) -> float:
        if trial.number == 5:
            trial.study.stop()
        return 1.0

    study = create_study()
    study.optimize(_objective, timeout=timeout, show_progress_bar=True)
    _, err = capsys.readouterr()
    assert expected in err


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_optimize_without_progbar_timeout(
    n_jobs: int, capsys: _pytest.capture.CaptureFixture
) -> None:

    study = create_study()
    study.optimize(lambda _: 1.0, timeout=2.0, n_jobs=n_jobs)
    _, err = capsys.readouterr()

    assert "00:02/00:02" not in err
    assert "100%" not in err


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_optimize_progbar_n_trials_prioritized(
    n_jobs: int, capsys: _pytest.capture.CaptureFixture
) -> None:

    study = create_study()
    study.optimize(lambda _: 1.0, n_trials=10, n_jobs=n_jobs, timeout=10.0, show_progress_bar=True)
    _, err = capsys.readouterr()

    assert "10/10" in err
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
    def _objective(trial: Trial) -> float:
        if trial.number == 5:
            trial.study.stop()
        return 1.0

    study = create_study()
    study.optimize(_objective, n_jobs=n_jobs, show_progress_bar=True)
    _, err = capsys.readouterr()

    # We can't simply test if stderr is empty, since we're not sure
    # what else could write to it. Instead, we are testing for a character
    # that forms progress bar borders.
    assert "|" not in err


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_optimize_without_progbar_no_constraints(
    n_jobs: int, capsys: _pytest.capture.CaptureFixture
) -> None:
    def _objective(trial: Trial) -> float:
        if trial.number == 5:
            trial.study.stop()
        return 1.0

    study = create_study()
    study.optimize(_objective, n_jobs=n_jobs)
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
    assert states == [TrialState.FAIL] * 10

    # If a trial is failed with an exception and the exception isn't caught by the study,
    # callbacks aren't invoked.
    states = []
    callbacks = [with_lock(lambda study, trial: states.append(trial.state))]
    with pytest.raises(ZeroDivisionError):
        study.optimize(lambda t: 1 / 0, callbacks=callbacks, n_trials=10, n_jobs=n_jobs, catch=())
    assert states == []


@pytest.mark.parametrize("n_objectives", [2, 3])
def test_optimize_with_multi_objectives(n_objectives: int) -> None:
    directions = ["minimize" for _ in range(n_objectives)]
    study = create_study(directions=directions)

    def objective(trial: Trial) -> List[float]:
        return [trial.suggest_float("v{}".format(i), 0, 5) for i in range(n_objectives)]

    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 10

    for trial in study.trials:
        assert trial.values
        assert len(trial.values) == n_objectives


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        frozen_trial = _optimize._run_trial(study, lambda _: float("inf"), catch=())
        assert frozen_trial.state == TrialState.COMPLETE
        assert frozen_trial.value == float("inf")


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_automatically_fail(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        frozen_trial = _optimize._run_trial(study, lambda _: float("nan"), catch=())
        assert frozen_trial.state == TrialState.FAIL
        assert frozen_trial.value is None

        frozen_trial = _optimize._run_trial(study, lambda _: None, catch=())  # type: ignore
        assert frozen_trial.state == TrialState.FAIL
        assert frozen_trial.value is None

        frozen_trial = _optimize._run_trial(study, lambda _: object(), catch=())  # type: ignore
        assert frozen_trial.state == TrialState.FAIL
        assert frozen_trial.value is None

        frozen_trial = _optimize._run_trial(study, lambda _: [0, 1], catch=())  # type: ignore
        assert frozen_trial.state == TrialState.FAIL
        assert frozen_trial.value is None


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_pruned(storage_mode: str) -> None:
    def gen_func(intermediate: Optional[float] = None) -> Callable[[Trial], float]:
        def func(trial: Trial) -> float:
            if intermediate is not None:
                trial.report(step=1, value=intermediate)
            raise TrialPruned

        return func

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        frozen_trial = _optimize._run_trial(study, gen_func(), catch=())
        assert frozen_trial.state == TrialState.PRUNED
        assert frozen_trial.value is None

        frozen_trial = _optimize._run_trial(study, gen_func(intermediate=1), catch=())
        assert frozen_trial.state == TrialState.PRUNED
        assert frozen_trial.value == 1

        frozen_trial = _optimize._run_trial(study, gen_func(intermediate=float("nan")), catch=())
        assert frozen_trial.state == TrialState.PRUNED
        assert frozen_trial.value is None


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_catch_exception(storage_mode: str) -> None:
    def func_value_error(_: Trial) -> float:
        raise ValueError

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        frozen_trial = _optimize._run_trial(study, func_value_error, catch=(ValueError,))
        assert frozen_trial.state == TrialState.FAIL


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_exception(storage_mode: str) -> None:
    def func_value_error(_: Trial) -> float:
        raise ValueError

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        with pytest.raises(ValueError):
            _optimize._run_trial(study, func_value_error, ())


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_optimize_invoke_tell_with_suppressing_warning(storage_mode: str) -> None:
    def func_numerical(trial: Trial) -> float:
        return trial.suggest_float("v", 0, 10)

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        with mock.patch(
            "optuna.study._optimize._tell_with_warning", side_effect=_tell_with_warning
        ) as mock_obj:
            _optimize._run_trial(study, func_numerical, ())
            mock_obj.assert_called_once_with(
                study=mock.ANY,
                trial=mock.ANY,
                values=mock.ANY,
                state=mock.ANY,
                suppress_warning=True,
            )
