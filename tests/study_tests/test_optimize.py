from unittest import mock

import pytest

from optuna import create_study
from optuna import Trial
from optuna.study import _optimize
from optuna.testing.storage import STORAGE_MODES
from optuna.testing.storage import StorageSupplier
from optuna.trial import TrialState


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_value_error(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        # Test trial with acceptable exception.
        def func_value_error(_: Trial) -> float:

            raise ValueError

        frozen_trial = _optimize._run_trial(study, func_value_error, catch=(ValueError,))

        assert frozen_trial.state == TrialState.FAIL

        # Test trial with unacceptable exception.
        with pytest.raises(ValueError):
            _optimize._run_trial(study, func_value_error, catch=(ArithmeticError,))


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_none(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        # Test trial with invalid objective value: None
        def func_none(_: Trial) -> float:

            return None  # type: ignore

        frozen_trial = _optimize._run_trial(study, func_none, catch=(Exception,))

        assert frozen_trial.state == TrialState.FAIL


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_nan(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        # Test trial with invalid objective value: nan
        def func_nan(_: Trial) -> float:

            return float("nan")

        frozen_trial = _optimize._run_trial(study, func_nan, catch=(Exception,))

        assert frozen_trial.state == TrialState.FAIL


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_nonnumerical(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        def func_nonnumerical(_: Trial) -> float:

            return "value"  # type: ignore

        frozen_trial = _optimize._run_trial(study, func_nonnumerical, catch=())

        assert frozen_trial.state == TrialState.FAIL


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_invoke_study_tell_with_suppressing_warning(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        def func_numerical(trial: Trial) -> float:
            return trial.suggest_float("v", 0, 10)

        study.tell = mock.MagicMock(side_effect=study.tell)  # type: ignore
        study._tell_with_warning = mock.MagicMock(  # type: ignore
            side_effect=study._tell_with_warning,
        )

        _optimize._run_trial(study, func_numerical, catch=())

        study.tell.assert_not_called()
        study._tell_with_warning.assert_called_with(
            mock.ANY, values=mock.ANY, state=mock.ANY, suppress_warning=True
        )
