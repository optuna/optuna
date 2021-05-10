from typing import List
from typing import Tuple

import pytest

import optuna
from optuna.trial import TrialState


def test_patient_pruner_patience() -> None:

    optuna.pruners.PatientPruner(None, 0)
    optuna.pruners.PatientPruner(None, 1)

    with pytest.raises(ValueError):
        optuna.pruners.PatientPruner(None, -1)


def test_patient_pruner_with_one_trial() -> None:

    pruner = optuna.pruners.PatientPruner(None, 0)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1, 1)

    # The pruner is not activated at a first trial.
    assert not trial.should_prune()


def test_patient_pruner_intermediate_values_nan() -> None:

    pruner = optuna.pruners.PatientPruner(None, 0, 0)
    study = optuna.study.create_study(pruner=pruner)

    trial = study.ask()

    # A pruner is not activated if a trial does not have any intermediate values.
    assert not trial.should_prune()

    trial.report(float("nan"), 1)
    # A pruner is not activated if a trial has only one intermediate value.
    assert not trial.should_prune()

    trial.report(float("nan"), 2)
    # A pruner is not activated if a trial has only nan in intermediate values.
    assert not trial.should_prune()

    trial.report(1.0, 3)
    # A pruner is not activated if a trial has only nan in intermediate values.
    assert not trial.should_prune()

    trial.report(float("nan"), 3)
    # A pruner is not activated if a trial has only nan in intermediate values.
    assert not trial.should_prune()
