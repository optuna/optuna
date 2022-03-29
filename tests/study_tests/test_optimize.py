from unittest import mock

import pytest

from optuna import create_study
from optuna import Trial
from optuna.study import _optimize
from optuna.study._tell import _tell_with_warning
from optuna.testing.storage import STORAGE_MODES
from optuna.testing.storage import StorageSupplier
from optuna.trial import TrialState


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_optimize_catch_exception(storage_mode: str) -> None:
    def func_value_error(_: Trial) -> float:
        raise ValueError

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        frozen_trial = _optimize._run_trial(study, func_value_error, catch=(ValueError,))
        assert frozen_trial.state == TrialState.FAIL


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_optimize_exception(storage_mode: str) -> None:
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
