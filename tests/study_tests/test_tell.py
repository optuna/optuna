from unittest.mock import patch

import pytest

from optuna import create_study
from optuna.testing.storage import StorageSupplier
from optuna.trial import TrialState


def test_tell() -> None:
    study = create_study()
    assert len(study.trials) == 0

    trial = study.ask()
    assert len(study.trials) == 1
    assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 0

    study.tell(trial, 1.0)
    assert len(study.trials) == 1
    assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 1

    study.tell(study.ask(), state=TrialState.PRUNED)
    assert len(study.trials) == 2
    assert len(study.get_trials(states=(TrialState.PRUNED,))) == 1

    study.tell(study.ask(), state=TrialState.FAIL)
    assert len(study.trials) == 3
    assert len(study.get_trials(states=(TrialState.FAIL,))) == 1

    with pytest.raises(ValueError):
        study.tell(study.ask(), state=TrialState.RUNNING)

    with pytest.raises(ValueError):
        study.tell(study.ask(), state=TrialState.WAITING)


def test_tell_trial_variations() -> None:
    study = create_study()

    study.tell(study.ask().number, 1.0)

    # Trial that has not been asked for cannot be told.
    with pytest.raises(ValueError):
        study.tell(study.ask().number + 1, 1.0)

    with pytest.raises(TypeError):
        study.tell("1", 1.0)  # type: ignore


def test_tell_duplicate_tell() -> None:
    study = create_study()

    trial = study.ask()
    study.tell(trial, 1.0)

    # Should not panic when passthrough is enabled.
    study.tell(trial, 1.0, skip_if_finished=True)

    with pytest.raises(RuntimeError):
        study.tell(trial, 1.0, skip_if_finished=False)


def test_tell_values() -> None:
    study = create_study()

    study.tell(study.ask(), 1.0)
    study.tell(study.ask(), [1.0])
    assert len(study.trials) == 2

    # Check invalid values, e.g. ones that cannot be cast to float.
    with pytest.warns(UserWarning):
        study.tell(study.ask(), "a")  # type: ignore
        assert len(study.trials) == 3
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    # Check number of values.
    with pytest.warns(UserWarning):
        study.tell(study.ask(), [])
        assert len(study.trials) == 4
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    with pytest.warns(UserWarning):
        study.tell(study.ask(), [1.0, 2.0])
        assert len(study.trials) == 5
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    study = create_study(directions=["minimize", "maximize"])
    study.tell(study.ask(), [1.0, 2.0])
    assert len(study.trials) == 1

    with pytest.warns(UserWarning):
        study.tell(study.ask(), [])
        assert len(study.trials) == 2
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    with pytest.warns(UserWarning):
        study.tell(study.ask(), [1.0])
        assert len(study.trials) == 3
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    with pytest.warns(UserWarning):
        study.tell(study.ask(), [1.0, 2.0, 3.0])
        assert len(study.trials) == 4
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    # Missing values for completions.
    with pytest.raises(ValueError):
        study.tell(study.ask(), state=TrialState.COMPLETE)

    # Either state or values is required.
    with pytest.warns(UserWarning):
        study.tell(study.ask())
        assert len(study.trials) == 6
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None


def test_tell_storage_not_implemented_trial_number() -> None:
    with StorageSupplier("inmemory") as storage:

        with patch.object(
            storage,
            "get_trial_id_from_study_id_trial_number",
            side_effect=NotImplementedError,
        ):
            study = create_study(storage=storage)

            study.tell(study.ask(), 1.0)

            # Storage missing implementation for method required to map trial numbers back to
            # trial IDs.
            with pytest.warns(UserWarning):
                study.tell(study.ask().number, 1.0)

            with pytest.raises(ValueError):
                study.tell(study.ask().number + 1, 1.0)


def test_tell_pruned_values() -> None:
    # See also `test_run_trial_with_trial_pruned`.
    study = create_study()

    trial = study.ask()

    trial.report(2.0, step=1)

    study.tell(trial, state=TrialState.PRUNED)
    assert study.trials[-1].value == 2.0

    trial = study.ask()

    study.tell(trial, state=TrialState.PRUNED)
    assert study.trials[-1].value is None

    trial = study.ask()

    trial.report(float("inf"), step=1)

    study.tell(trial, state=TrialState.PRUNED)
    assert study.trials[-1].value == float("inf")
    assert study.trials[-1].state == TrialState.PRUNED

    trial = study.ask()

    trial.report(float("nan"), step=1)

    study.tell(trial, state=TrialState.PRUNED)
    assert study.trials[-1].value is None
    assert study.trials[-1].state == TrialState.PRUNED
