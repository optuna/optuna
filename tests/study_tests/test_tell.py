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

    study.tell(study.ask(), [1.0])
    assert len(study.trials) == 2
    assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 2

    # `trial` could be int.
    study.tell(study.ask().number, 1.0)
    assert len(study.trials) == 3
    assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 3

    # Inf is supported as values.
    study.tell(study.ask(), float("inf"))
    assert len(study.trials) == 4
    assert len(study.get_trials(states=(TrialState.COMPLETE,))) == 4

    study.tell(study.ask(), state=TrialState.PRUNED)
    assert len(study.trials) == 5
    assert len(study.get_trials(states=(TrialState.PRUNED,))) == 1

    study.tell(study.ask(), state=TrialState.FAIL)
    assert len(study.trials) == 6
    assert len(study.get_trials(states=(TrialState.FAIL,))) == 1


def test_tell_pruned() -> None:
    study = create_study()

    study.tell(study.ask(), state=TrialState.PRUNED)
    assert study.trials[-1].value is None
    assert study.trials[-1].state == TrialState.PRUNED

    # Store the last intermediates as value.
    trial = study.ask()
    trial.report(2.0, step=1)
    study.tell(trial, state=TrialState.PRUNED)
    assert study.trials[-1].value == 2.0
    assert study.trials[-1].state == TrialState.PRUNED

    # Inf is also supported as a value.
    trial = study.ask()
    trial.report(float("inf"), step=1)
    study.tell(trial, state=TrialState.PRUNED)
    assert study.trials[-1].value == float("inf")
    assert study.trials[-1].state == TrialState.PRUNED

    # NaN is not supported as a value.
    trial = study.ask()
    trial.report(float("nan"), step=1)
    study.tell(trial, state=TrialState.PRUNED)
    assert study.trials[-1].value is None
    assert study.trials[-1].state == TrialState.PRUNED


def test_tell_automatically_fail() -> None:
    study = create_study()

    # Check invalid values, e.g. str cannot be cast to float.
    with pytest.warns(UserWarning):
        study.tell(study.ask(), "a")  # type: ignore
        assert len(study.trials) == 1
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    # Check invalid values, e.g. `None` that cannot be cast to float.
    with pytest.warns(UserWarning):
        study.tell(study.ask(), None)  # type: ignore
        assert len(study.trials) == 2
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    # Check number of values.
    with pytest.warns(UserWarning):
        study.tell(study.ask(), [])
        assert len(study.trials) == 3
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    # Check wrong number of values, e.g. two values for single direction.
    with pytest.warns(UserWarning):
        study.tell(study.ask(), [1.0, 2.0])
        assert len(study.trials) == 4
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    # Both state and values are not specified.
    with pytest.warns(UserWarning):
        study.tell(study.ask())
        assert len(study.trials) == 5
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    # Nan is not supported.
    with pytest.warns(UserWarning):
        study.tell(study.ask(), float("nan"))
        assert len(study.trials) == 6
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None


def test_tell_multi_objective() -> None:
    study = create_study(directions=["minimize", "maximize"])
    study.tell(study.ask(), [1.0, 2.0])
    assert len(study.trials) == 1


def test_tell_multi_objective_automatically_fail() -> None:
    # Number of values doesn't match the length of directions.
    study = create_study(directions=["minimize", "maximize"])

    with pytest.warns(UserWarning):
        study.tell(study.ask(), [])
        assert len(study.trials) == 1
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    with pytest.warns(UserWarning):
        study.tell(study.ask(), [1.0])
        assert len(study.trials) == 2
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None

    with pytest.warns(UserWarning):
        study.tell(study.ask(), [1.0, 2.0, 3.0])
        assert len(study.trials) == 3
        assert study.trials[-1].state == TrialState.FAIL
        assert study.trials[-1].values is None


def test_tell_invalid() -> None:
    study = create_study()

    # Missing values for completions.
    with pytest.raises(ValueError):
        study.tell(study.ask(), state=TrialState.COMPLETE)

    # `state` must be None or finished state
    with pytest.raises(ValueError):
        study.tell(study.ask(), state=TrialState.RUNNING)

    # `state` must be None or finished state
    with pytest.raises(ValueError):
        study.tell(study.ask(), state=TrialState.WAITING)

    # Trial that has not been asked for cannot be told.
    with pytest.raises(ValueError):
        study.tell(study.ask().number + 1, 1.0)

    # It must be Trial or int for trial.
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
