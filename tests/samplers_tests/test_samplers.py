import math
import numpy as np
import pytest
import typing  # NOQA

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.samplers import BaseSampler  # NOQA

if optuna.types.TYPE_CHECKING:
    from optuna.trial import T  # NOQA

parametrize_sampler = pytest.mark.parametrize(
    'sampler_class', [optuna.samplers.RandomSampler, optuna.samplers.TPESampler])


@parametrize_sampler
@pytest.mark.parametrize('distribution', [UniformDistribution(-1., 1.),
                                          UniformDistribution(0., 1.),
                                          UniformDistribution(-1., 0.)])
def test_uniform(sampler_class, distribution):
    # type: (typing.Callable[[], BaseSampler], UniformDistribution) -> None

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        return trial.suggest_uniform('x', distribution.low, distribution.high)

    def check(study):
        # type: (optuna.study.Study) -> None

        points = np.array([study.sampler.sample(study.storage, study.study_id, 'x', distribution)
                           for _ in range(100)])
        assert np.all(points >= distribution.low)
        assert np.all(points < distribution.high)

    _study = optuna.study.create_study(sampler=sampler_class())

    check(_study)

    # Execute optimization for samplers which have startup trials such as TPESampler.
    _study.optimize(objective, n_trials=10)

    check(_study)


@parametrize_sampler
@pytest.mark.parametrize('distribution', [LogUniformDistribution(1e-7, 1.)])
def test_log_uniform(sampler_class, distribution):
    # type: (typing.Callable[[], BaseSampler], LogUniformDistribution) -> None

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        return trial.suggest_loguniform('x', distribution.low, distribution.high)

    def check(study):
        # type: (optuna.study.Study) -> None

        points = np.array([study.sampler.sample(study.storage, study.study_id, 'x', distribution)
                           for _ in range(100)])
        assert np.all(points >= distribution.low)
        assert np.all(points < distribution.high)

    _study = optuna.study.create_study(sampler=sampler_class())

    check(_study)

    # Execute optimization for samplers which have startup trials such as TPESampler.
    _study.optimize(objective, n_trials=10)

    check(_study)


@parametrize_sampler
@pytest.mark.parametrize('distribution', [DiscreteUniformDistribution(-10, 10, 1),
                                          DiscreteUniformDistribution(-10.2, 10.2, 0.1),
                                          DiscreteUniformDistribution(64., 1312., 160.),
                                          DiscreteUniformDistribution(0, 10, math.pi)])
def test_discrete_uniform(sampler_class, distribution):
    # type: (typing.Callable[[], BaseSampler], DiscreteUniformDistribution) -> None

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        return trial.suggest_discrete_uniform('x', distribution.low, distribution.high,
                                              distribution.q)

    def check(study):
        # type: (optuna.study.Study) -> None

        points = np.array([study.sampler.sample(study.storage, study.study_id, 'x', distribution)
                           for _ in range(100)])
        assert np.all(points >= distribution.low)
        assert np.all(points <= distribution.high)

        # Check all points are multiples of distribution.q except endpoints.
        points = points[points != distribution.low]
        points = points[points != distribution.high]
        points -= distribution.low
        points /= distribution.q
        round_points = np.round(points)
        np.testing.assert_almost_equal(round_points, points)

    _study = optuna.study.create_study(sampler=sampler_class())

    check(_study)

    # Execute optimization for samplers which have startup trials such as TPESampler.
    _study.optimize(objective, n_trials=10)

    check(_study)


@parametrize_sampler
@pytest.mark.parametrize('distribution', [IntUniformDistribution(-10, 10),
                                          IntUniformDistribution(0, 10),
                                          IntUniformDistribution(-10, 0)])
def test_int(sampler_class, distribution):
    # type: (typing.Callable[[], BaseSampler], IntUniformDistribution) -> None

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        return trial.suggest_int('x', distribution.low, distribution.high)

    def check(study):
        # type: (optuna.study.Study) -> None

        points = np.array([study.sampler.sample(study.storage, study.study_id, 'x', distribution)
                           for _ in range(100)])
        assert np.all(points >= distribution.low)
        assert np.all(points <= distribution.high)

    _study = optuna.study.create_study(sampler=sampler_class())

    check(_study)

    # Execute optimization for samplers which have startup trials such as TPESampler.
    _study.optimize(objective, n_trials=10)

    check(_study)


@parametrize_sampler
@pytest.mark.parametrize('choices', [(1, 2, 3),
                                     ('a', 'b', 'c')])
def test_categorical(sampler_class, choices):
    # type: (typing.Callable[[], BaseSampler], typing.Tuple[T, ...]) -> None

    distribution = CategoricalDistribution(choices)

    def objective(trial):
        # type: (optuna.trial.Trial) -> float

        trial.suggest_categorical('x', choices)
        return 1.0

    def check(study):
        # type: (optuna.study.Study) -> None

        points = np.array([study.sampler.sample(study.storage, study.study_id, 'x', distribution)
                           for _ in range(100)])
        # 'x' value is corresponding to an index of distribution.choices.
        assert np.all(points >= 0)
        assert np.all(points <= len(distribution.choices) - 1)
        round_points = np.round(points)
        np.testing.assert_almost_equal(round_points, points)

    _study = optuna.study.create_study(sampler=sampler_class())

    check(_study)

    # Execute optimization for samplers which have startup trials such as TPESampler.
    _study.optimize(objective, n_trials=10)

    check(_study)
