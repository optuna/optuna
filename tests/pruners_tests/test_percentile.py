import math

import pytest

import optuna
from optuna.pruners import percentile
from optuna.study import StudyDirection
from optuna.trial import TrialState
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    from typing import List  # NOQA
    from typing import Tuple  # NOQA

    from optuna.study import Study  # NOQA


def test_percentile_pruner_percentile():
    # type: () -> None

    optuna.pruners.PercentilePruner(0.0)
    optuna.pruners.PercentilePruner(25.0)
    optuna.pruners.PercentilePruner(100.0)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(-0.1)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(100.1)


def test_percentile_pruner_n_startup_trials():
    # type: () -> None

    optuna.pruners.PercentilePruner(25.0, n_startup_trials=0)
    optuna.pruners.PercentilePruner(25.0, n_startup_trials=5)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(25.0, n_startup_trials=-1)


def test_percentile_pruner_n_warmup_steps():
    # type: () -> None

    optuna.pruners.PercentilePruner(25.0, n_warmup_steps=0)
    optuna.pruners.PercentilePruner(25.0, n_warmup_steps=5)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(25.0, n_warmup_steps=-1)


def test_percentile_pruner_interval_steps():
    # type: () -> None

    optuna.pruners.PercentilePruner(25.0, interval_steps=1)
    optuna.pruners.PercentilePruner(25.0, interval_steps=5)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(25.0, interval_steps=-1)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(25.0, interval_steps=0)


def test_percentile_pruner_with_one_trial():
    # type: () -> None

    study = optuna.study.create_study()
    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(1, 1)
    pruner = optuna.pruners.PercentilePruner(25.0, 0, 0)

    # A pruner is not activated at a first trial.
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))


@pytest.mark.parametrize(
    "direction_value", [("minimize", [1, 2, 3, 4, 5], 2.1), ("maximize", [1, 2, 3, 4, 5], 3.9),]
)
def test_25_percentile_pruner_intermediate_values(direction_value):
    # type: (Tuple[str, List[float], float]) -> None

    direction, intermediate_values, latest_value = direction_value
    pruner = optuna.pruners.PercentilePruner(25.0, 0, 0)
    study = optuna.study.create_study(direction=direction)

    for v in intermediate_values:
        trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
        trial.report(v, 1)
        study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    # A pruner is not activated if a trial has no intermediate values.
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(latest_value, 1)
    # A pruner is activated if a trial has an intermediate value.
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))


def test_25_percentile_pruner_intermediate_values_nan():
    # type: () -> None

    pruner = optuna.pruners.PercentilePruner(25.0, 0, 0)
    study = optuna.study.create_study()

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(float("nan"), 1)
    # A pruner is not activated if the study does not have any previous trials.
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(float("nan"), 1)
    # A pruner is activated if the best intermediate value of this trial is NaN.
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(1, 1)
    # A pruner is not activated if the 25 percentile intermediate value is NaN.
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))


@pytest.mark.parametrize(
    "direction_expected", [(StudyDirection.MINIMIZE, 0.1), (StudyDirection.MAXIMIZE, 0.2)]
)
def test_get_best_intermediate_result_over_steps(direction_expected):
    # type: (Tuple[StudyDirection, float]) -> None

    direction, expected = direction_expected

    if direction == StudyDirection.MINIMIZE:
        study = optuna.study.create_study(direction="minimize")
    else:
        study = optuna.study.create_study(direction="maximize")

    # FrozenTrial.intermediate_values has no elements.
    trial_id_empty = study._storage.create_new_trial(study._study_id)
    trial_empty = study._storage.get_trial(trial_id_empty)

    with pytest.raises(ValueError):
        percentile._get_best_intermediate_result_over_steps(trial_empty, direction)

    # Input value has no NaNs but float values.
    trial_id_float = study._storage.create_new_trial(study._study_id)
    trial_float = optuna.trial.Trial(study, trial_id_float)
    trial_float.report(0.1, step=0)
    trial_float.report(0.2, step=1)
    frozen_trial_float = study._storage.get_trial(trial_id_float)
    assert expected == percentile._get_best_intermediate_result_over_steps(
        frozen_trial_float, direction
    )

    # Input value has a float value and a NaN.
    trial_id_float_nan = study._storage.create_new_trial(study._study_id)
    trial_float_nan = optuna.trial.Trial(study, trial_id_float_nan)
    trial_float_nan.report(0.3, step=0)
    trial_float_nan.report(float("nan"), step=1)
    frozen_trial_float_nan = study._storage.get_trial(trial_id_float_nan)
    assert 0.3 == percentile._get_best_intermediate_result_over_steps(
        frozen_trial_float_nan, direction
    )

    # Input value has a NaN only.
    trial_id_nan = study._storage.create_new_trial(study._study_id)
    trial_nan = optuna.trial.Trial(study, trial_id_nan)
    trial_nan.report(float("nan"), step=0)
    frozen_trial_nan = study._storage.get_trial(trial_id_nan)
    assert math.isnan(
        percentile._get_best_intermediate_result_over_steps(frozen_trial_nan, direction)
    )


def test_get_percentile_intermediate_result_over_trials():
    # type: () -> None

    def setup_study(trial_num, _intermediate_values):
        # type: (int, List[List[float]]) -> Study

        _study = optuna.study.create_study(direction="minimize")
        trial_ids = [_study._storage.create_new_trial(_study._study_id) for _ in range(trial_num)]

        for step, values in enumerate(_intermediate_values):
            # Study does not have any trials.
            with pytest.raises(ValueError):
                _all_trials = _study._storage.get_all_trials(_study._study_id)
                _direction = _study._storage.get_study_direction(_study._study_id)
                percentile._get_percentile_intermediate_result_over_trials(
                    _all_trials, _direction, step, 25
                )

            for i in range(trial_num):
                trial_id = trial_ids[i]
                value = values[i]
                _study._storage.set_trial_intermediate_value(trial_id, step, value)

        # Set trial states complete because this method ignores incomplete trials.
        for trial_id in trial_ids:
            _study._storage.set_trial_state(trial_id, TrialState.COMPLETE)

        return _study

    # Input value has no NaNs but float values (step=0).
    intermediate_values = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    study = setup_study(9, intermediate_values)
    all_trials = study._storage.get_all_trials(study._study_id)
    direction = study._storage.get_study_direction(study._study_id)
    assert 0.3 == percentile._get_percentile_intermediate_result_over_trials(
        all_trials, direction, 0, 25.0
    )

    # Input value has a float value and NaNs (step=1).
    intermediate_values.append(
        [0.1, 0.2, 0.3, 0.4, 0.5, float("nan"), float("nan"), float("nan"), float("nan")]
    )
    study = setup_study(9, intermediate_values)
    all_trials = study._storage.get_all_trials(study._study_id)
    direction = study._storage.get_study_direction(study._study_id)
    assert 0.2 == percentile._get_percentile_intermediate_result_over_trials(
        all_trials, direction, 1, 25.0
    )

    # Input value has NaNs only (step=2).
    intermediate_values.append(
        [
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        ]
    )
    study = setup_study(9, intermediate_values)
    all_trials = study._storage.get_all_trials(study._study_id)
    direction = study._storage.get_study_direction(study._study_id)
    assert math.isnan(
        percentile._get_percentile_intermediate_result_over_trials(all_trials, direction, 2, 75)
    )
