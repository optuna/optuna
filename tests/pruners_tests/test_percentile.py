import math
import pytest

import optuna
from optuna.pruners.percentile import get_best_intermediate_result_over_steps
from optuna.structs import StudyDirection
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


@pytest.mark.parametrize('direction_expected', [(StudyDirection.MINIMIZE, 0.1),
                                                (StudyDirection.MAXIMIZE, 0.2)])
def test_get_best_intermediate_result_over_steps(direction_expected):
    # type: (Tuple[StudyDirection, float]) -> None

    direction, expected = direction_expected

    if direction == StudyDirection.MINIMIZE:
        study = optuna.study.create_study(direction="minimize")
    else:
        study = optuna.study.create_study(direction="maximize")

    # FrozenTrial.intermediate_values has no elements.
    trial_id_empty = study.storage.create_new_trial_id(study.study_id)
    trial_empty = study.storage.get_trial(trial_id_empty)

    with pytest.raises(ValueError):
        get_best_intermediate_result_over_steps(trial_empty, direction)

    # Input value has no NaNs but float values.
    trial_id_float = study.storage.create_new_trial_id(study.study_id)
    trial_float = optuna.trial.Trial(study, trial_id_float)
    trial_float.report(0.1, step=0)
    trial_float.report(0.2, step=1)
    frozen_trial_float = study.storage.get_trial(trial_id_float)
    assert expected == get_best_intermediate_result_over_steps(frozen_trial_float, direction)

    # Input value has a float value and a NaN.
    trial_id_float_nan = study.storage.create_new_trial_id(study.study_id)
    trial_float_nan = optuna.trial.Trial(study, trial_id_float_nan)
    trial_float_nan.report(0.3, step=0)
    trial_float_nan.report(float('nan'), step=1)
    frozen_trial_float_nan = study.storage.get_trial(trial_id_float_nan)
    assert 0.3 == get_best_intermediate_result_over_steps(frozen_trial_float_nan, direction)

    # Input value has a NaN only.
    trial_id_nan = study.storage.create_new_trial_id(study.study_id)
    trial_nan = optuna.trial.Trial(study, trial_id_nan)
    trial_nan.report(float('nan'), step=0)
    frozen_trial_nan = study.storage.get_trial(trial_id_nan)
    assert math.isnan(get_best_intermediate_result_over_steps(frozen_trial_nan, direction))
