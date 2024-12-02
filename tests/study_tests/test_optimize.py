from __future__ import annotations

from collections.abc import Callable
from collections.abc import Generator
from unittest import mock

from _pytest.logging import LogCaptureFixture
import pytest

from optuna import create_study
from optuna import logging
from optuna import Trial
from optuna import TrialPruned
from optuna.study import _optimize
from optuna.study._tell import _tell_with_warning
from optuna.study._tell import STUDY_TELL_WARNING_KEY
from optuna.testing.objectives import fail_objective
from optuna.testing.storages import STORAGE_MODES
from optuna.testing.storages import StorageSupplier
from optuna.trial import TrialState


@pytest.fixture(autouse=True)
def logging_setup() -> Generator[None, None, None]:
    # We need to reconstruct our default handler to properly capture stderr.
    logging._reset_library_root_logger()
    logging.enable_default_handler()
    logging.set_verbosity(logging.INFO)
    logging.enable_propagation()

    yield

    # After testing, restore default propagation setting.
    logging.disable_propagation()


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial(storage_mode: str, caplog: LogCaptureFixture) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        caplog.clear()
        frozen_trial = _optimize._run_trial(study, lambda _: 1.0, catch=())
        assert frozen_trial.state == TrialState.COMPLETE
        assert frozen_trial.value == 1.0
        assert "Trial 0 finished with value: 1.0 and parameters" in caplog.text

        caplog.clear()
        frozen_trial = _optimize._run_trial(study, lambda _: float("inf"), catch=())
        assert frozen_trial.state == TrialState.COMPLETE
        assert frozen_trial.value == float("inf")
        assert "Trial 1 finished with value: inf and parameters" in caplog.text

        caplog.clear()
        frozen_trial = _optimize._run_trial(study, lambda _: -float("inf"), catch=())
        assert frozen_trial.state == TrialState.COMPLETE
        assert frozen_trial.value == -float("inf")
        assert "Trial 2 finished with value: -inf and parameters" in caplog.text


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_automatically_fail(storage_mode: str, caplog: LogCaptureFixture) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        frozen_trial = _optimize._run_trial(study, lambda _: float("nan"), catch=())
        assert frozen_trial.state == TrialState.FAIL
        assert frozen_trial.value is None

        frozen_trial = _optimize._run_trial(study, lambda _: None, catch=())  # type: ignore[arg-type,return-value] # noqa: E501
        assert frozen_trial.state == TrialState.FAIL
        assert frozen_trial.value is None

        frozen_trial = _optimize._run_trial(study, lambda _: object(), catch=())  # type: ignore[arg-type,return-value] # noqa: E501
        assert frozen_trial.state == TrialState.FAIL
        assert frozen_trial.value is None

        frozen_trial = _optimize._run_trial(study, lambda _: [0, 1], catch=())
        assert frozen_trial.state == TrialState.FAIL
        assert frozen_trial.value is None


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_pruned(storage_mode: str, caplog: LogCaptureFixture) -> None:
    def gen_func(intermediate: float | None = None) -> Callable[[Trial], float]:
        def func(trial: Trial) -> float:
            if intermediate is not None:
                trial.report(step=1, value=intermediate)
            raise TrialPruned

        return func

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        caplog.clear()
        frozen_trial = _optimize._run_trial(study, gen_func(), catch=())
        assert frozen_trial.state == TrialState.PRUNED
        assert frozen_trial.value is None
        assert "Trial 0 pruned." in caplog.text

        caplog.clear()
        frozen_trial = _optimize._run_trial(study, gen_func(intermediate=1), catch=())
        assert frozen_trial.state == TrialState.PRUNED
        assert frozen_trial.value == 1
        assert "Trial 1 pruned." in caplog.text

        caplog.clear()
        frozen_trial = _optimize._run_trial(study, gen_func(intermediate=float("nan")), catch=())
        assert frozen_trial.state == TrialState.PRUNED
        assert frozen_trial.value is None
        assert "Trial 2 pruned." in caplog.text


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_catch_exception(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        frozen_trial = _optimize._run_trial(study, fail_objective, catch=(ValueError,))
        assert frozen_trial.state == TrialState.FAIL
        assert STUDY_TELL_WARNING_KEY not in frozen_trial.system_attrs


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_exception(storage_mode: str) -> None:
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        with pytest.raises(ValueError):
            _optimize._run_trial(study, fail_objective, ())

    # Test trial with unacceptable exception.
    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        with pytest.raises(ValueError):
            _optimize._run_trial(study, fail_objective, (ArithmeticError,))


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_invoke_tell_with_suppressing_warning(storage_mode: str) -> None:
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
                value_or_values=mock.ANY,
                state=mock.ANY,
                suppress_warning=True,
            )
