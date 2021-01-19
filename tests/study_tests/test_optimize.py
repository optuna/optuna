from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import pytest

from optuna import _optimize
from optuna import create_study
from optuna import Study
from optuna import Trial
from optuna import TrialPruned
from optuna.exceptions import TrialPruned as TrialPruned_in_exceptions
from optuna.structs import TrialPruned as TrialPruned_in_structs
from optuna.testing.storage import STORAGE_MODES
from optuna.testing.storage import StorageSupplier
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


def func(trial: Trial, x_max: float = 1.0) -> float:

    x = trial.suggest_uniform("x", -x_max, x_max)
    y = trial.suggest_loguniform("y", 20, 30)
    z = trial.suggest_categorical("z", (-1.0, 1.0))
    assert isinstance(z, float)
    return (x - 2) ** 2 + (y - 25) ** 2 + z


def check_params(params: Dict[str, Any]) -> None:

    assert sorted(params.keys()) == ["x", "y", "z"]


def check_value(value: Optional[float]) -> None:

    assert isinstance(value, float)
    assert -1.0 <= value <= 12.0 ** 2 + 5.0 ** 2 + 1.0


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


@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_run_trial(storage_mode: str) -> None:

    with StorageSupplier(storage_mode) as storage:
        study = create_study(storage=storage)

        # Test trial without exception.
        _optimize._run_trial(study, func, catch=(Exception,))
        check_study(study)

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
