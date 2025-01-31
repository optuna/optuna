from __future__ import annotations

import math
import warnings

import pytest

import optuna
from optuna.pruners import _percentile
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import TrialState


def test_percentile_pruner_percentile() -> None:
    optuna.pruners.PercentilePruner(0.0)
    optuna.pruners.PercentilePruner(25.0)
    optuna.pruners.PercentilePruner(100.0)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(-0.1)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(100.1)


def test_percentile_pruner_n_startup_trials() -> None:
    optuna.pruners.PercentilePruner(25.0, n_startup_trials=0)
    optuna.pruners.PercentilePruner(25.0, n_startup_trials=5)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(25.0, n_startup_trials=-1)


def test_percentile_pruner_n_warmup_steps() -> None:
    optuna.pruners.PercentilePruner(25.0, n_warmup_steps=0)
    optuna.pruners.PercentilePruner(25.0, n_warmup_steps=5)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(25.0, n_warmup_steps=-1)


def test_percentile_pruner_interval_steps() -> None:
    optuna.pruners.PercentilePruner(25.0, interval_steps=1)
    optuna.pruners.PercentilePruner(25.0, interval_steps=5)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(25.0, interval_steps=-1)

    with pytest.raises(ValueError):
        optuna.pruners.PercentilePruner(25.0, interval_steps=0)


def test_percentile_pruner_with_one_trial() -> None:
    pruner = optuna.pruners.PercentilePruner(25.0, 0, 0)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1, 1)

    # A pruner is not activated at a first trial.
    assert not trial.should_prune()


@pytest.mark.parametrize(
    "direction_value", [("minimize", [1, 2, 3, 4, 5], 2.1), ("maximize", [1, 2, 3, 4, 5], 3.9)]
)
def test_25_percentile_pruner_intermediate_values(
    direction_value: tuple[str, list[float], float],
) -> None:
    direction, intermediate_values, latest_value = direction_value
    pruner = optuna.pruners.PercentilePruner(25.0, 0, 0)
    study = optuna.study.create_study(direction=direction, pruner=pruner)

    for v in intermediate_values:
        trial = study.ask()
        trial.report(v, 1)
        study.tell(trial, v)

    trial = study.ask()
    # A pruner is not activated if a trial has no intermediate values.
    assert not trial.should_prune()

    trial.report(latest_value, 1)
    # A pruner is activated if a trial has an intermediate value.
    assert trial.should_prune()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_25_percentile_pruner_intermediate_values_nan() -> None:
    pruner = optuna.pruners.PercentilePruner(25.0, 0, 0)
    study = optuna.study.create_study(pruner=pruner)

    trial = study.ask()
    trial.report(float("nan"), 1)
    # A pruner is not activated if the study does not have any previous trials.
    assert not trial.should_prune()
    study.tell(trial, -1)

    trial = study.ask()
    trial.report(float("nan"), 1)
    # A pruner is activated if the best intermediate value of this trial is NaN.
    assert trial.should_prune()
    study.tell(trial, -1)

    trial = study.ask()
    trial.report(1, 1)
    # A pruner is not activated if the 25 percentile intermediate value is NaN.
    assert not trial.should_prune()


@pytest.mark.parametrize(
    "direction_expected", [(StudyDirection.MINIMIZE, 0.1), (StudyDirection.MAXIMIZE, 0.2)]
)
def test_get_best_intermediate_result_over_steps(
    direction_expected: tuple[StudyDirection, float],
) -> None:
    direction, expected = direction_expected

    if direction == StudyDirection.MINIMIZE:
        study = optuna.study.create_study(direction="minimize")
    else:
        study = optuna.study.create_study(direction="maximize")

    # FrozenTrial.intermediate_values has no elements.
    trial_id_empty = study._storage.create_new_trial(study._study_id)
    trial_empty = study._storage.get_trial(trial_id_empty)

    with pytest.raises(ValueError):
        _percentile._get_best_intermediate_result_over_steps(trial_empty, direction)

    # Input value has no NaNs but float values.
    trial_float = study.ask()
    trial_float.report(0.1, step=0)
    trial_float.report(0.2, step=1)
    frozen_trial_float = study._storage.get_trial(trial_float._trial_id)
    assert expected == _percentile._get_best_intermediate_result_over_steps(
        frozen_trial_float, direction
    )

    # Input value has a float value and a NaN.
    trial_float_nan = study.ask()
    trial_float_nan.report(0.3, step=0)
    trial_float_nan.report(float("nan"), step=1)
    frozen_trial_float_nan = study._storage.get_trial(trial_float_nan._trial_id)
    assert 0.3 == _percentile._get_best_intermediate_result_over_steps(
        frozen_trial_float_nan, direction
    )

    # Input value has a NaN only.
    trial_nan = study.ask()
    trial_nan.report(float("nan"), step=0)
    frozen_trial_nan = study._storage.get_trial(trial_nan._trial_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        assert math.isnan(
            _percentile._get_best_intermediate_result_over_steps(frozen_trial_nan, direction)
        )


def test_get_percentile_intermediate_result_over_trials() -> None:
    def setup_study(trial_num: int, _intermediate_values: list[list[float]]) -> Study:
        _study = optuna.study.create_study(direction="minimize")
        trial_ids = [_study._storage.create_new_trial(_study._study_id) for _ in range(trial_num)]

        for step, values in enumerate(_intermediate_values):
            # Study does not have any complete trials.
            with pytest.raises(ValueError):
                completed_trials = _study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
                _direction = _study.direction
                _percentile._get_percentile_intermediate_result_over_trials(
                    completed_trials, _direction, step, 25, 1
                )

            for i in range(trial_num):
                trial_id = trial_ids[i]
                value = values[i]
                _study._storage.set_trial_intermediate_value(trial_id, step, value)

        # Set trial states complete because this method ignores incomplete trials.
        for trial_id in trial_ids:
            _study._storage.set_trial_state_values(trial_id, state=TrialState.COMPLETE)

        return _study

    # Input value has no NaNs but float values (step=0).
    intermediate_values = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    study = setup_study(9, intermediate_values)
    all_trials = study.get_trials()
    direction = study.direction
    assert 0.3 == _percentile._get_percentile_intermediate_result_over_trials(
        all_trials, direction, 0, 25.0, 1
    )

    # Input value has a float value and NaNs (step=1).
    intermediate_values.append(
        [0.1, 0.2, 0.3, 0.4, 0.5, float("nan"), float("nan"), float("nan"), float("nan")]
    )
    study = setup_study(9, intermediate_values)
    all_trials = study.get_trials()
    direction = study.direction
    assert 0.2 == _percentile._get_percentile_intermediate_result_over_trials(
        all_trials, direction, 1, 25.0, 1
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
    all_trials = study.get_trials()
    direction = study.direction
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        assert math.isnan(
            _percentile._get_percentile_intermediate_result_over_trials(
                all_trials, direction, 2, 75, 1
            )
        )

        # n_min_trials = 2.
        assert math.isnan(
            _percentile._get_percentile_intermediate_result_over_trials(
                all_trials, direction, 2, 75, 2
            )
        )
