from typing import Callable
from typing import Optional
from unittest import mock

import pytest

from optuna import create_study
from optuna import Trial
from optuna import TrialPruned
from optuna.study import _optimize
from optuna.study._tell import _tell_with_warning
from optuna.testing.storage import STORAGE_MODES
from optuna.testing.storage import StorageSupplier
from optuna.trial import TrialState


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
