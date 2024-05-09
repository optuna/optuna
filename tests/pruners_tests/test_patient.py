from __future__ import annotations

import warnings

import pytest

import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.study import StudyDirection


def test_patient_pruner_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        optuna.pruners.PatientPruner(None, 0)


def test_patient_pruner_patience() -> None:
    optuna.pruners.PatientPruner(None, 0)
    optuna.pruners.PatientPruner(None, 1)

    with pytest.raises(ValueError):
        optuna.pruners.PatientPruner(None, -1)


def test_patient_pruner_min_delta() -> None:
    optuna.pruners.PatientPruner(None, 0, 0.0)
    optuna.pruners.PatientPruner(None, 0, 1.0)

    with pytest.raises(ValueError):
        optuna.pruners.PatientPruner(None, 0, -1)


def test_patient_pruner_with_one_trial() -> None:
    pruner = optuna.pruners.PatientPruner(None, 0)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1, 0)

    # The pruner is not activated at a first trial.
    assert not trial.should_prune()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_patient_pruner_intermediate_values_nan() -> None:
    pruner = optuna.pruners.PatientPruner(None, 0, 0)
    study = optuna.study.create_study(pruner=pruner)

    trial = study.ask()

    # A pruner is not activated if a trial does not have any intermediate values.
    assert not trial.should_prune()

    trial.report(float("nan"), 0)
    # A pruner is not activated if a trial has only one intermediate value.
    assert not trial.should_prune()

    trial.report(1.0, 1)
    # A pruner is not activated if a trial has only nan in intermediate values.
    assert not trial.should_prune()

    trial.report(float("nan"), 2)
    # A pruner is not activated if a trial has only nan in intermediate values.
    assert not trial.should_prune()


@pytest.mark.parametrize(
    "patience,min_delta,direction,intermediates,expected_prune_steps",
    [
        (0, 0, "maximize", [1, 0], [1]),
        (1, 0, "maximize", [2, 1, 0], [2]),
        (0, 0, "minimize", [0, 1], [1]),
        (1, 0, "minimize", [0, 1, 2], [2]),
        (0, 1.0, "maximize", [1, 0], []),
        (1, 1.0, "maximize", [3, 2, 1, 0], [3]),
        (0, 1.0, "minimize", [0, 1], []),
        (1, 1.0, "minimize", [0, 1, 2, 3], [3]),
    ],
)
def test_patient_pruner_intermediate_values(
    patience: int,
    min_delta: float,
    direction: str,
    intermediates: list[int],
    expected_prune_steps: list[int],
) -> None:
    pruner = optuna.pruners.PatientPruner(None, patience, min_delta)
    study = optuna.study.create_study(pruner=pruner, direction=direction)

    trial = study.ask()

    pruned = []
    for step, value in enumerate(intermediates):
        trial.report(value, step)
        if trial.should_prune():
            pruned.append(step)
    assert pruned == expected_prune_steps


def test_patient_pruner_direction() -> None:
    optuna.pruners.PatientPruner(None, patience=0, direction="minimize")
    optuna.pruners.PatientPruner(None, patience=0, direction="maximize")
    optuna.pruners.PatientPruner(None, patience=0, direction=StudyDirection.MINIMIZE)
    optuna.pruners.PatientPruner(None, patience=0, direction=StudyDirection.MAXIMIZE)
    optuna.pruners.PatientPruner(None, patience=0, direction=None)

    with pytest.raises(ValueError):
        optuna.pruners.PatientPruner(None, patience=0, direction="1")


def test_patient_pruner_requires_direction_if_multi_objective() -> None:
    pruner = optuna.pruners.PatientPruner(None, patience=0)
    study = optuna.study.create_study(directions=["minimize", "minimize"], pruner=pruner)
    trial = study.ask()
    trial.report(1, 1)
    trial.report(1, 2)
    study.tell(trial, (1, 1))
    trial = study.ask()
    trial.report(1, 1)
    trial.report(1, 2)

    # The direction of pruner must be set for multi-objective optimization.
    with pytest.raises(ValueError):
        trial.should_prune()


def test_patient_pruner_ignores_direction_if_single_objective() -> None:
    pruner = optuna.pruners.PatientPruner(None, patience=0, direction="minimize")
    study = optuna.study.create_study(directions=["minimize"], pruner=pruner)
    trial = study.ask()
    trial.report(1, 1)
    trial.report(1, 2)
    study.tell(trial, 1)
    trial = study.ask()
    trial.report(1, 1)
    trial.report(1, 2)

    # The direction of pruner should not be set for single-objective optimization.
    with pytest.warns(UserWarning):
        trial.should_prune()


@pytest.mark.parametrize("intermediate_expected", [(0, False), (2, True)])
def test_patient_pruner_works_if_multi_objective(intermediate_expected: tuple[int, bool]) -> None:
    pruner = optuna.pruners.PatientPruner(None, patience=0, direction="minimize")
    study = optuna.study.create_study(directions=["minimize", "minimize"], pruner=pruner)
    trial = study.ask()
    trial.report(1, 1)
    trial.report(1, 2)
    study.tell(trial, (1, 1))
    trial = study.ask()
    trial.report(1, 1)
    trial.report(intermediate_expected[0], 2)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("ignore", ExperimentalWarning)
        assert trial.should_prune() == intermediate_expected[1]
        assert len(w) == 0
