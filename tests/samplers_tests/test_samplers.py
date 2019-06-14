import numpy as np
import pytest

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.samplers import BaseSampler
from optuna.study import InTrialStudy

if optuna.types.TYPE_CHECKING:
    import typing  # NOQA
    from typing import Dict  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.structs import FrozenTrial  # NOQA
    from optuna.study import Study  # NOQA
    from optuna.trial import T  # NOQA
    from optuna.trial import Trial  # NOQA

parametrize_sampler = pytest.mark.parametrize(
    'sampler_class',
    [optuna.samplers.RandomSampler, lambda: optuna.samplers.TPESampler(n_startup_trials=0)])


@parametrize_sampler
@pytest.mark.parametrize(
    'distribution',
    [UniformDistribution(-1., 1.),
     UniformDistribution(0., 1.),
     UniformDistribution(-1., 0.)])
def test_uniform(sampler_class, distribution):
    # type: (typing.Callable[[], BaseSampler], UniformDistribution) -> None

    study = optuna.study.create_study(sampler=sampler_class())
    in_trial_study = InTrialStudy(study)
    points = np.array([
        study.sampler.sample_independent(in_trial_study, _create_new_trial(study), 'x',
                                         distribution) for _ in range(100)
    ])
    assert np.all(points >= distribution.low)
    assert np.all(points < distribution.high)
    assert not isinstance(
        study.sampler.sample_independent(in_trial_study, _create_new_trial(study), 'x',
                                         distribution), np.floating)


@parametrize_sampler
@pytest.mark.parametrize('distribution', [LogUniformDistribution(1e-7, 1.)])
def test_log_uniform(sampler_class, distribution):
    # type: (typing.Callable[[], BaseSampler], LogUniformDistribution) -> None

    study = optuna.study.create_study(sampler=sampler_class())
    in_trial_study = InTrialStudy(study)
    points = np.array([
        study.sampler.sample_independent(in_trial_study, _create_new_trial(study), 'x',
                                         distribution) for _ in range(100)
    ])
    assert np.all(points >= distribution.low)
    assert np.all(points < distribution.high)
    assert not isinstance(
        study.sampler.sample_independent(in_trial_study, _create_new_trial(study), 'x',
                                         distribution), np.floating)


@parametrize_sampler
@pytest.mark.parametrize(
    'distribution',
    [DiscreteUniformDistribution(-10, 10, 0.1),
     DiscreteUniformDistribution(-10.2, 10.2, 0.1)])
def test_discrete_uniform(sampler_class, distribution):
    # type: (typing.Callable[[], BaseSampler], DiscreteUniformDistribution) -> None

    study = optuna.study.create_study(sampler=sampler_class())
    in_trial_study = InTrialStudy(study)
    points = np.array([
        study.sampler.sample_independent(in_trial_study, _create_new_trial(study), 'x',
                                         distribution) for _ in range(100)
    ])
    assert np.all(points >= distribution.low)
    assert np.all(points <= distribution.high)
    assert not isinstance(
        study.sampler.sample_independent(in_trial_study, _create_new_trial(study), 'x',
                                         distribution), np.floating)

    # Check all points are multiples of distribution.q.
    points = points
    points -= distribution.low
    points /= distribution.q
    round_points = np.round(points)
    np.testing.assert_almost_equal(round_points, points)


@parametrize_sampler
@pytest.mark.parametrize('distribution', [
    IntUniformDistribution(-10, 10),
    IntUniformDistribution(0, 10),
    IntUniformDistribution(-10, 0)
])
def test_int(sampler_class, distribution):
    # type: (typing.Callable[[], BaseSampler], IntUniformDistribution) -> None

    study = optuna.study.create_study(sampler=sampler_class())
    in_trial_study = InTrialStudy(study)
    points = np.array([
        study.sampler.sample_independent(in_trial_study, _create_new_trial(study), 'x',
                                         distribution) for _ in range(100)
    ])
    assert np.all(points >= distribution.low)
    assert np.all(points <= distribution.high)
    assert not isinstance(
        study.sampler.sample_independent(in_trial_study, _create_new_trial(study), 'x',
                                         distribution), np.integer)


@parametrize_sampler
@pytest.mark.parametrize('choices', [(1, 2, 3), ('a', 'b', 'c')])
def test_categorical(sampler_class, choices):
    # type: (typing.Callable[[], BaseSampler], typing.Tuple[T, ...]) -> None

    distribution = CategoricalDistribution(choices)

    study = optuna.study.create_study(sampler=sampler_class())
    in_trial_study = InTrialStudy(study)
    points = np.array([
        study.sampler.sample_independent(in_trial_study, _create_new_trial(study), 'x',
                                         distribution) for _ in range(100)
    ])
    # 'x' value is corresponding to an index of distribution.choices.
    assert np.all(points >= 0)
    assert np.all(points <= len(distribution.choices) - 1)
    round_points = np.round(points)
    np.testing.assert_almost_equal(round_points, points)


def _create_new_trial(study):
    # type: (Study) -> FrozenTrial

    trial_id = study.storage.create_new_trial_id(study.study_id)
    return study.storage.get_trial(trial_id)


class FixedSampler(BaseSampler):
    def __init__(self, relative_search_space, relative_params, unknown_param_value):
        # type: (Dict[str, BaseDistribution], Dict[str, float], float) -> None

        self.relative_search_space = relative_search_space
        self.relative_params = relative_params
        self.unknown_param_value = unknown_param_value

    def infer_relative_search_space(self, study, trial):
        # type: (InTrialStudy, FrozenTrial) -> Dict[str, BaseDistribution]

        return self.relative_search_space

    def sample_relative(self, study, trial, search_space):
        # type: (InTrialStudy, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, float]

        return self.relative_params

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (InTrialStudy, FrozenTrial, str, BaseDistribution) -> float

        return self.unknown_param_value


def test_sample_relative():
    # type: () -> None

    relative_search_space = {
        'a': UniformDistribution(low=0, high=5),
        'b': CategoricalDistribution(choices=('foo', 'bar', 'baz')),
        'c': IntUniformDistribution(low=20, high=50),  # Not exist in `relative_params`.
    }
    relative_params = {
        'a': 3.2,
        'b': 2,
    }
    unknown_param_value = 30

    sampler = FixedSampler(  # type: ignore
        relative_search_space, relative_params, unknown_param_value)
    study = optuna.study.create_study(sampler=sampler)

    def objective(trial):
        # type: (Trial) -> float

        # Predefined parameters are sampled by `sample_relative()` method.
        assert trial.suggest_uniform('a', 0, 5) == 3.2
        assert trial.suggest_categorical('b', ['foo', 'bar', 'baz']) == 'baz'

        # Other parameters are sampled by `sample_independent()` method.
        assert trial.suggest_int('c', 20, 50) == unknown_param_value
        assert trial.suggest_loguniform('d', 1, 100) == unknown_param_value
        assert trial.suggest_uniform('e', 20, 40) == unknown_param_value

        return 0.0

    study.optimize(objective, n_trials=10, catch=())
    for trial in study.trials:
        assert trial.params == {'a': 3.2, 'b': 'baz', 'c': 30, 'd': 30, 'e': 30}
