from __future__ import annotations

import pytest

import optuna


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
