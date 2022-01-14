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

        trial = _optimize._run_trial(study, func_value_error, catch=(ValueError,))
        frozen_trial = study._storage.get_trial(trial._trial_id)

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

        trial = _optimize._run_trial(study, func_none, catch=(Exception,))
        frozen_trial = study._storage.get_trial(trial._trial_id)

        assert frozen_trial.state == TrialState.FAIL


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_nan(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        # Test trial with invalid objective value: nan
        def func_nan(_: Trial) -> float:

            return float("nan")

        trial = _optimize._run_trial(study, func_nan, catch=(Exception,))
        frozen_trial = study._storage.get_trial(trial._trial_id)

        assert frozen_trial.state == TrialState.FAIL


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial_nonnumerical(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        def func_nonnumerical(_: Trial) -> float:

            return "value"  # type: ignore

        trial = _optimize._run_trial(study, func_nonnumerical, catch=())
        frozen_trial = study._storage.get_trial(trial._trial_id)

        assert frozen_trial.state == TrialState.FAIL
