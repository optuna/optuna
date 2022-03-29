from unittest import mock

import pytest

from optuna import create_study
from optuna import Study
from optuna import Trial
from optuna.study._tell import _tell_with_warning
from optuna.testing.storage import STORAGE_MODES
from optuna.testing.storage import StorageSupplier
from optuna.trial import TrialState


N_TRIALS = 1


def check_trial_state(study: Study, state: TrialState) -> None:
    assert len(study.trials) == N_TRIALS
    assert study.trials[0].state == state


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_optimize_catch_exception(storage_mode: str) -> None:

    def func_value_error(_: Trial) -> float:
        raise ValueError

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(func_value_error, catch=(ValueError,), n_trials=N_TRIALS)
        check_trial_state(study, TrialState.FAIL)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_optimize_exception(storage_mode: str) -> None:

    def func_value_error(_: Trial) -> float:
        raise ValueError

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        with pytest.raises(ValueError):
            study.optimize(func_value_error, n_trials=N_TRIALS)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_optimize_none(storage_mode: str) -> None:

    def func_none(_: Trial) -> float:
        return None  # type: ignore

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(func_none, n_trials=N_TRIALS)
        check_trial_state(study, TrialState.FAIL)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_optimize_nan(storage_mode: str) -> None:

    def func_nan(_: Trial) -> float:
        return float("nan")

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(func_nan, n_trials=N_TRIALS)
        check_trial_state(study, TrialState.FAIL)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_optimize_nonnumerical(storage_mode: str) -> None:

    def func_nonnumerical(_: Trial) -> float:
        return "value"  # type: ignore

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)
        study.optimize(func_nonnumerical, n_trials=N_TRIALS)
        check_trial_state(study, TrialState.FAIL)


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_optimize_invoke_tell_with_suppressing_warning(storage_mode: str) -> None:

    def func_numerical(trial: Trial) -> float:
        return trial.suggest_float("v", 0, 10)

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        with mock.patch(
            "optuna.study._optimize._tell_with_warning", side_effect=_tell_with_warning
        ) as mock_obj:
            study.tell = mock.MagicMock(side_effect=study.tell)  # type: ignore
            study.optimize(func_numerical, n_trials=N_TRIALS)
            mock_obj.assert_called_once_with(
                study=mock.ANY,
                trial=mock.ANY,
                values=mock.ANY,
                state=mock.ANY,
                suppress_warning=True,
            )
