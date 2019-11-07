import pytest

import optuna
from optuna.structs import TrialState
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import Tuple  # NOQA


def test_median_pruner_with_one_trial():
    # type: () -> None

    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study.study_id))
    trial.report(1, 1)
    pruner = optuna.pruners.MedianPruner(0, 0)

    # A pruner is not activated at a first trial.
    assert not pruner.prune(
        study=study, trial=study._storage.get_trial(trial.trial_id))


@pytest.mark.parametrize('direction_value', [('minimize', 2), ('maximize', 0.5)])
def test_median_pruner_intermediate_values(direction_value):
    # type: (Tuple[str, float]) -> None

    direction, intermediate_value = direction_value
    pruner = optuna.pruners.MedianPruner(0, 0)
    study = optuna.study.create_study(direction=direction)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study.study_id))
    trial.report(1, 1)
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study.study_id))
    # A pruner is not activated if a trial has no intermediate values.
    assert not pruner.prune(
        study=study, trial=study._storage.get_trial(trial.trial_id))

    trial.report(intermediate_value, 1)
    # A pruner is activated if a trial has an intermediate value.
    assert pruner.prune(
        study=study, trial=study._storage.get_trial(trial.trial_id))


def test_median_pruner_intermediate_values_nan():
    # type: () -> None

    pruner = optuna.pruners.MedianPruner(0, 0)
    study = optuna.study.create_study()

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study.study_id))
    trial.report(float('nan'), 1)
    # A pruner is not activated if the study does not have any previous trials.
    assert not pruner.prune(
        study=study, trial=study._storage.get_trial(trial.trial_id))
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study.study_id))
    trial.report(float('nan'), 1)
    # A pruner is activated if the best intermediate value of this trial is NaN.
    assert pruner.prune(
        study=study, trial=study._storage.get_trial(trial.trial_id))
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study.study_id))
    trial.report(1, 1)
    # A pruner is not activated if the median intermediate value is NaN.
    assert not pruner.prune(
        study=study, trial=study._storage.get_trial(trial.trial_id))


def test_median_pruner_n_startup_trials():
    # type: () -> None

    pruner = optuna.pruners.MedianPruner(2, 0)
    study = optuna.study.create_study()

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study.study_id))
    trial.report(1, 1)
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study.study_id))
    trial.report(2, 1)
    # A pruner is not activated during startup trials.
    assert not pruner.prune(
        study=study, trial=study._storage.get_trial(trial.trial_id))
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study.study_id))
    trial.report(3, 1)
    # A pruner is activated after startup trials.
    assert pruner.prune(
        study=study, trial=study._storage.get_trial(trial.trial_id))


def test_median_pruner_n_warmup_steps():
    # type: () -> None

    pruner = optuna.pruners.MedianPruner(0, 1)
    study = optuna.study.create_study()

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study.study_id))
    trial.report(1, 1)
    trial.report(1, 2)
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study.study_id))
    trial.report(2, 1)
    # A pruner is not activated during warm-up steps.
    assert not pruner.prune(
        study=study, trial=study._storage.get_trial(trial.trial_id))

    trial.report(2, 2)
    # A pruner is activated after warm-up steps.
    assert pruner.prune(
        study=study, trial=study._storage.get_trial(trial.trial_id))


@pytest.mark.parametrize(
    'n_warmup_steps,interval_steps,report_steps,expected_prune_steps', [
        (1, 2, 1, (3,)),
        (0, 3, 10, (10, 20, 30)),
        (2, 3, 10, (10, 20, 30)),
        (19, 3, 10, (19, 20, 30)),
        (0, 10, 3, (3, 12, 21, 30)),
        (2, 10, 3, (3, 12, 24, 33)),
        (11, 10, 3, (12, 21, 33)),
    ])
def test_median_pruner_interval_steps(
        n_warmup_steps, interval_steps, report_steps, expected_prune_steps):
    # type: (int, int, int, Tuple[int, ...]) -> None

    pruner = optuna.pruners.MedianPruner(0, n_warmup_steps, interval_steps)
    study = optuna.study.create_study()

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study.study_id))
    n_steps = max(expected_prune_steps)
    for i in range(1, 1 + n_steps):
        trial.report(1, i)
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study.study_id))
    for i in range(1, 1 + n_steps):
        if i % report_steps == 0:
            trial.report(2, i)
        assert (pruner.prune(study=study, trial=study._storage.get_trial(trial.trial_id))
                == (i > n_warmup_steps and i in expected_prune_steps))
