import pytest

import optuna
from optuna.structs import TrialState
from optuna import types

if types.TYPE_CHECKING:
    from typing import List  # NOQA
    from typing import Tuple  # NOQA


def test_percentile_pruner_with_one_trial():
    # type: () -> None

    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(1, 1)
    pruner = optuna.pruners.PercentilePruner(25.0, 0, 0)

    # A pruner is not activated at a first trial.
    assert not pruner.prune(
        storage=study.storage, study_id=study.study_id, trial_id=trial._trial_id, step=1)


@pytest.mark.parametrize('direction_value', [
    ('minimize', [1, 2, 3, 4, 5], 2.1),
    ('maximize', [1, 2, 3, 4, 5], 3.9),
])
def test_25_percentile_pruner_intermediate_values(direction_value):
    # type: (Tuple[str, List[float], float]) -> None

    direction, intermediate_values, latest_value = direction_value
    pruner = optuna.pruners.PercentilePruner(25.0, 0, 0)
    study = optuna.study.create_study(direction=direction)

    for v in intermediate_values:
        trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
        trial.report(v, 1)
        study.storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    # A pruner is not activated if a trial has no intermediate values.
    assert not pruner.prune(
        storage=study.storage, study_id=study.study_id, trial_id=trial._trial_id, step=1)

    trial.report(latest_value, 1)
    # A pruner is activated if a trial has an intermediate value.
    assert pruner.prune(
        storage=study.storage, study_id=study.study_id, trial_id=trial._trial_id, step=1)


def test_25_percentile_pruner_intermediate_values_nan():
    # type: () -> None

    pruner = optuna.pruners.PercentilePruner(25.0, 0, 0)
    study = optuna.study.create_study()

    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(float('nan'), 1)
    # A pruner is not activated if the study does not have any previous trials.
    assert not pruner.prune(
        storage=study.storage, study_id=study.study_id, trial_id=trial._trial_id, step=1)
    study.storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(float('nan'), 1)
    # A pruner is activated if the best intermediate value of this trial is NaN.
    assert pruner.prune(
        storage=study.storage, study_id=study.study_id, trial_id=trial._trial_id, step=1)
    study.storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study.storage.create_new_trial_id(study.study_id))
    trial.report(1, 1)
    # A pruner is not activated if the 25 percentile intermediate value is NaN.
    assert not pruner.prune(
        storage=study.storage, study_id=study.study_id, trial_id=trial._trial_id, step=1)
