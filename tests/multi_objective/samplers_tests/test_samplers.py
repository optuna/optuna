from typing import Callable
from typing import Sequence

import numpy as np
import pytest

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna import multi_objective
from optuna.multi_objective.samplers import BaseMultiObjectiveSampler
from optuna.multi_objective.trial import CategoricalChoiceType

parametrize_sampler = pytest.mark.parametrize(
    "sampler_class", [optuna.multi_objective.samplers.RandomMultiObjectiveSampler,],
)


@parametrize_sampler
@pytest.mark.parametrize(
    "distribution",
    [
        UniformDistribution(-1.0, 1.0),
        UniformDistribution(0.0, 1.0),
        UniformDistribution(-1.0, 0.0),
    ],
)
def test_uniform(
    sampler_class: Callable[[], BaseMultiObjectiveSampler], distribution: UniformDistribution
) -> None:
    study = optuna.multi_objective.study.create_study(
        ["minimize", "maximize"], sampler=sampler_class()
    )
    points = np.array(
        [
            study.sampler.sample_independent(study, _create_new_trial(study), "x", distribution)
            for _ in range(100)
        ]
    )
    assert np.all(points >= distribution.low)
    assert np.all(points < distribution.high)
    assert not isinstance(
        study.sampler.sample_independent(study, _create_new_trial(study), "x", distribution),
        np.floating,
    )


@parametrize_sampler
@pytest.mark.parametrize("distribution", [LogUniformDistribution(1e-7, 1.0)])
def test_log_uniform(
    sampler_class: Callable[[], BaseMultiObjectiveSampler], distribution: LogUniformDistribution
) -> None:
    study = optuna.multi_objective.study.create_study(
        ["minimize", "maximize"], sampler=sampler_class()
    )
    points = np.array(
        [
            study.sampler.sample_independent(study, _create_new_trial(study), "x", distribution)
            for _ in range(100)
        ]
    )
    assert np.all(points >= distribution.low)
    assert np.all(points < distribution.high)
    assert not isinstance(
        study.sampler.sample_independent(study, _create_new_trial(study), "x", distribution),
        np.floating,
    )


@parametrize_sampler
@pytest.mark.parametrize(
    "distribution",
    [DiscreteUniformDistribution(-10, 10, 0.1), DiscreteUniformDistribution(-10.2, 10.2, 0.1)],
)
def test_discrete_uniform(
    sampler_class: Callable[[], BaseMultiObjectiveSampler],
    distribution: DiscreteUniformDistribution,
) -> None:
    study = optuna.multi_objective.study.create_study(
        ["minimize", "maximize"], sampler=sampler_class()
    )
    points = np.array(
        [
            study.sampler.sample_independent(study, _create_new_trial(study), "x", distribution)
            for _ in range(100)
        ]
    )
    assert np.all(points >= distribution.low)
    assert np.all(points <= distribution.high)
    assert not isinstance(
        study.sampler.sample_independent(study, _create_new_trial(study), "x", distribution),
        np.floating,
    )

    # Check all points are multiples of distribution.q.
    points = points
    points -= distribution.low
    points /= distribution.q
    round_points = np.round(points)
    np.testing.assert_almost_equal(round_points, points)


@parametrize_sampler
@pytest.mark.parametrize(
    "distribution",
    [
        IntUniformDistribution(-10, 10),
        IntUniformDistribution(0, 10),
        IntUniformDistribution(-10, 0),
        IntUniformDistribution(-10, 10, 2),
        IntUniformDistribution(0, 10, 2),
        IntUniformDistribution(-10, 0, 2),
    ],
)
def test_int(
    sampler_class: Callable[[], BaseMultiObjectiveSampler], distribution: IntUniformDistribution
) -> None:
    study = optuna.multi_objective.study.create_study(
        ["minimize", "maximize"], sampler=sampler_class()
    )
    points = np.array(
        [
            study.sampler.sample_independent(study, _create_new_trial(study), "x", distribution)
            for _ in range(100)
        ]
    )
    assert np.all(points >= distribution.low)
    assert np.all(points <= distribution.high)
    assert not isinstance(
        study.sampler.sample_independent(study, _create_new_trial(study), "x", distribution),
        np.integer,
    )


@parametrize_sampler
@pytest.mark.parametrize("choices", [(1, 2, 3), ("a", "b", "c"), (1, "a")])
def test_categorical(
    sampler_class: Callable[[], BaseMultiObjectiveSampler],
    choices: Sequence[CategoricalChoiceType],
) -> None:
    distribution = CategoricalDistribution(choices)

    study = optuna.multi_objective.study.create_study(
        ["minimize", "maximize"], sampler=sampler_class()
    )

    def sample() -> float:
        trial = _create_new_trial(study)
        param_value = study.sampler.sample_independent(study, trial, "x", distribution)
        return distribution.to_internal_repr(param_value)

    points = np.array([sample() for _ in range(100)])

    # 'x' value is corresponding to an index of distribution.choices.
    assert np.all(points >= 0)
    assert np.all(points <= len(distribution.choices) - 1)
    round_points = np.round(points)
    np.testing.assert_almost_equal(round_points, points)


def _create_new_trial(
    study: multi_objective.study.MultiObjectiveStudy,
) -> multi_objective.trial.FrozenMultiObjectiveTrial:
    trial_id = study._study._storage.create_new_trial(study._study._study_id)
    trial = study._study._storage.get_trial(trial_id)
    return multi_objective.trial.FrozenMultiObjectiveTrial(study.n_objectives, trial)
