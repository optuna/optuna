from typing import Callable
from typing import Optional

import pytest

from optuna import _optimize
from optuna import create_study
from optuna import Trial
from optuna import TrialPruned
from optuna.exceptions import TrialPruned as TrialPruned_in_exceptions
from optuna.structs import TrialPruned as TrialPruned_in_structs
from optuna.testing.storage import STORAGE_MODES
from optuna.testing.storage import StorageSupplier
from optuna.trial import TrialState


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
