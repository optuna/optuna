from __future__ import annotations

from typing import Literal
from unittest import mock

import pytest

import optuna
from optuna.testing.pruners import DeterministicPruner


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


@pytest.mark.parametrize("direction", ["minimize", "maximize"])
@pytest.mark.parametrize(
    "patience,intermediates",
    [
        (0, [float("nan"), 1.0]),
        (0, [1.0, float("nan")]),
        (0, [float("nan"), float("nan")]),
        (1, [float("nan"), float("nan"), 1.0, 1.0]),
        (1, [1.0, 1.0, float("nan"), float("nan")]),
        (1, [float("nan")] * 4),
    ],
)
def test_patient_pruner_intermediate_values_nan(
    direction: Literal["minimize", "maximize"], patience: int, intermediates: list[float]
) -> None:
    pruner = optuna.pruners.PatientPruner(None, patience, 0)
    study = optuna.study.create_study(pruner=pruner, direction=direction)

    trial = study.ask()

    # A pruner is not activated if a trial does not have any intermediate values.
    assert not trial.should_prune()

    for step, value in enumerate(intermediates):
        trial.report(value, step)
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
        (1, 0, "maximize", [float("nan"), 1, float("nan"), 0], [3]),
        (1, 0, "minimize", [float("nan"), 0, float("nan"), 1], [3]),
        (1, 1.0, "maximize", [float("nan"), 1, float("nan"), 0], []),
        (1, 1.0, "minimize", [float("nan"), 0, float("nan"), 1], []),
    ],
)
def test_patient_pruner_intermediate_values(
    patience: int,
    min_delta: float,
    direction: Literal["minimize", "maximize"],
    intermediates: list[float],
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


@pytest.mark.parametrize("is_pruning", [False, True])
@pytest.mark.parametrize(
    "intermediates,expected_calls",
    [
        ([], 0),
        ([1.0], 0),
        ([float("nan"), 1.0], 0),
        ([1.0, float("nan")], 0),
        ([float("nan"), float("nan")], 0),
        ([2.0, 1.0], 0),
        ([1.0, 2.0], 1),
    ],
)
def test_patient_pruner_wrapped_pruner(
    is_pruning: bool, intermediates: list[float], expected_calls: int
) -> None:
    wrapped_pruner = DeterministicPruner(is_pruning)
    pruner = optuna.pruners.PatientPruner(wrapped_pruner, 0)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    for step, value in enumerate(intermediates):
        trial.report(value, step)

    with mock.patch.object(wrapped_pruner, "prune", wraps=wrapped_pruner.prune) as prune:
        assert trial.should_prune() == (is_pruning and expected_calls == 1)
        assert prune.call_count == expected_calls
