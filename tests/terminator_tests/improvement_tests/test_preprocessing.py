from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.study._study_direction import StudyDirection
from optuna.terminator.improvement import _preprocessing
from optuna.trial import create_trial
from optuna.trial import FrozenTrial


def _get_trial_values(trials: list[FrozenTrial]) -> list[float]:
    values = []
    for t in trials:
        assert t.value is not None
        values.append(t.value)
    return values


@pytest.mark.parametrize("direction", (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE))
def test_preprocessing_pipeline(direction: StudyDirection) -> None:
    p1 = _preprocessing.NullPreprocessing()
    p2 = _preprocessing.NullPreprocessing()

    pipeline = _preprocessing.PreprocessingPipeline([p1, p2])

    with mock.patch.object(p1, "apply") as a1:
        with mock.patch.object(p2, "apply") as a2:
            pipeline.apply([], direction)

            a1.assert_called_once()
            a2.assert_called_once()


@pytest.mark.parametrize("direction", (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE))
def test_null_preprocessing(direction: StudyDirection) -> None:
    trials_before = [
        create_trial(params={"x": 0.0}, distributions={"x": FloatDistribution(-1, 1)}, value=1.0)
    ]
    p = _preprocessing.NullPreprocessing()
    trials_after = p.apply(trials_before, direction)
    assert trials_before == trials_after


@pytest.mark.parametrize("direction", (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE))
def test_unscale_log(direction: StudyDirection) -> None:
    trials_before = [
        create_trial(
            params={
                "categorical": 0,
                "float": 1,
                "float_log": 2,
                "int": 3,
                "int_log": 4,
            },
            distributions={
                "categorical": CategoricalDistribution((0, 1, 2)),
                "float": FloatDistribution(1, 10),
                "float_log": FloatDistribution(1, 10, log=True),
                "int": IntDistribution(1, 10),
                "int_log": IntDistribution(1, 10, log=True),
            },
            value=1.0,
        ),
    ]
    p = _preprocessing.UnscaleLog()
    trials_after = p.apply(trials_before, direction)

    assert len(trials_before) == len(trials_after) == 1
    assert trials_before[0].value == trials_after[0].value

    # Assert that the trial has no log distribution.
    for d in trials_after[0].distributions:
        if isinstance(d, (IntDistribution, FloatDistribution)):
            assert not d.log

    # Assert that the parameters are identical for the non-log distributions.
    assert trials_before[0].params["float"] == trials_after[0].params["float"]
    assert trials_before[0].params["int"] == trials_after[0].params["int"]
    assert trials_before[0].params["categorical"] == trials_after[0].params["categorical"]

    # Assert that the parameters are unscaled for the log distributions.
    assert trials_before[0].params["float_log"] == np.exp(trials_after[0].params["float_log"])
    assert trials_before[0].params["int_log"] == np.exp(trials_after[0].params["int_log"])


@pytest.mark.parametrize("direction", (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE))
def test_select_top_trials(direction: StudyDirection) -> None:
    values = [1, 0, 0, 2, 1]
    trials_before = [create_trial(value=v) for v in values]

    values_in_order = sorted(values, reverse=(direction == StudyDirection.MAXIMIZE))

    # Scenario: `n_trials` * `top_trials_ratio` is less than `min_n_trials`.
    top_trials_ratio = 0.4
    min_n_trials = 3
    p = _preprocessing.SelectTopTrials(
        top_trials_ratio=top_trials_ratio,
        min_n_trials=min_n_trials,
    )
    trials_after = p.apply(trials_before, direction)

    n_trials_expected = min_n_trials
    assert len(trials_after) == n_trials_expected
    assert _get_trial_values(trials_after) == values_in_order[:n_trials_expected]

    # Scenario: `n_trials` * `top_trials_ratio` is greater than `min_n_trials`.
    top_trials_ratio = 0.8
    min_n_trials = 3
    p = _preprocessing.SelectTopTrials(
        top_trials_ratio=top_trials_ratio,
        min_n_trials=min_n_trials,
    )
    trials_after = p.apply(trials_before, direction)

    n_trials_expected = int(len(values) * top_trials_ratio)
    assert len(trials_after) == n_trials_expected
    assert _get_trial_values(trials_after) == values_in_order[:n_trials_expected]


@pytest.mark.parametrize("direction", (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE))
def test_to_minimize(direction: StudyDirection) -> None:
    values = [1, 0, 0, 2, 1]
    trials_before = [create_trial(value=v) for v in values]

    expected_values = values[:]
    if direction == StudyDirection.MAXIMIZE:
        expected_values = [-1 * v for v in expected_values]

    p = _preprocessing.ToMinimize()
    trials_after = p.apply(trials_before, direction)
    assert _get_trial_values(trials_after) == expected_values


@pytest.mark.parametrize("direction", (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE))
def test_one_to_hot(direction: StudyDirection) -> None:
    # Scenario: categorical parameters are mapped to the float distribution.
    trials_before = [
        create_trial(
            params={"categorical": 0},
            distributions={"categorical": CategoricalDistribution((0, 1))},
            value=1.0,
        ),
        create_trial(
            params={"categorical": 1},
            distributions={"categorical": CategoricalDistribution((0, 1))},
            value=1.0,
        ),
    ]
    p = _preprocessing.OneToHot()
    trials_after = p.apply(trials_before, direction)

    for t in trials_after:
        # Distributions are expected to be mapped to `FloatDistribution(0, 1)` for each choice.
        assert len(t.distributions) == 2
        for d in t.distributions.values():
            assert d == FloatDistribution(0, 1)

    assert trials_after[0].params == {"i0_categorical": 1.0, "i1_categorical": 0.0}
    assert trials_after[1].params == {"i0_categorical": 0.0, "i1_categorical": 1.0}

    # Scenario: int and float parameters are identical even after the preprocessing.
    trials_before = [
        create_trial(
            params={
                "float": 1,
                "int": 3,
            },
            distributions={
                "float": FloatDistribution(1, 10),
                "int": IntDistribution(1, 10),
            },
            value=1.0,
        ),
    ]
    p = _preprocessing.OneToHot()
    trials_after = p.apply(trials_before, direction)

    for name_before, param_before in trials_before[0].params.items():
        distribution_before = trials_before[0].distributions[name_before]

        # Note that the param/distribution names are modified with "i0_" prefix to be consistent
        # with categorical parameters.
        name_after = f"i0_{name_before}"
        param_after = trials_after[0].params[name_after]
        distribution_after = trials_after[0].distributions[name_after]

        assert distribution_before == distribution_after
        assert param_before == param_after


@pytest.mark.parametrize("direction", (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE))
def test_add_random_inputs(direction: StudyDirection) -> None:
    n_additional_trials = 3
    dummy_value = -1
    distributions = {
        "bacon": CategoricalDistribution((0, 1, 2)),
        "egg": FloatDistribution(1, 10),
        "spam": IntDistribution(1, 10),
    }

    trials_before = [
        create_trial(
            params={"bacon": 0, "egg": 1, "spam": 10},
            distributions=distributions,
            value=1.0,
        ),
    ]
    p = _preprocessing.AddRandomInputs(
        n_additional_trials=n_additional_trials,
        dummy_value=dummy_value,
    )
    trials_after = p.apply(trials_before, direction)

    assert trials_before[0] == trials_after[0]
    assert len(trials_after) == len(trials_before) + n_additional_trials
    for t in trials_after[1:]:
        assert t.value == dummy_value
        assert t.distributions == distributions
        assert set(t.params.keys()) == set(distributions.keys())

        for name, distribution in distributions.items():
            assert distribution._contains(t.params[name])
