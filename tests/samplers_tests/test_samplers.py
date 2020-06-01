from collections import OrderedDict
import pickle

import numpy as np
import pytest

import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.samplers import BaseSampler
from optuna.testing.storage import StorageSupplier

if optuna.type_checking.TYPE_CHECKING:
    import typing  # NOQA
    from typing import Any  # NOQA
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA
    from typing import Sequence  # NOQA

    from optuna.distributions import BaseDistribution  # NOQA
    from optuna.distributions import CategoricalChoiceType  # NOQA
    from optuna.trial import FrozenTrial  # NOQA
    from optuna.study import Study  # NOQA
    from optuna.trial import Trial  # NOQA

parametrize_sampler = pytest.mark.parametrize(
    "sampler_class",
    [
        optuna.samplers.RandomSampler,
        lambda: optuna.samplers.TPESampler(n_startup_trials=0),
        lambda: optuna.integration.SkoptSampler(skopt_kwargs={"n_initial_points": 1}),
        lambda: optuna.integration.CmaEsSampler(),
    ],
)


@pytest.mark.parametrize("seed", [None, 0, 169208])
def test_pickle_random_sampler(seed):
    # type: (Optional[int]) -> None

    sampler = optuna.samplers.RandomSampler(seed)
    restored_sampler = pickle.loads(pickle.dumps(sampler))
    assert sampler._rng.bytes(10) == restored_sampler._rng.bytes(10)


def test_random_sampler_reseed_rng() -> None:
    sampler = optuna.samplers.RandomSampler()
    original_seed = sampler._rng.seed

    sampler.reseed_rng()
    assert original_seed != sampler._rng.seed


@parametrize_sampler
@pytest.mark.parametrize(
    "distribution",
    [
        UniformDistribution(-1.0, 1.0),
        UniformDistribution(0.0, 1.0),
        UniformDistribution(-1.0, 0.0),
    ],
)
def test_uniform(sampler_class, distribution):
    # type: (typing.Callable[[], BaseSampler], UniformDistribution) -> None

    study = optuna.study.create_study(sampler=sampler_class())
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
def test_log_uniform(sampler_class, distribution):
    # type: (typing.Callable[[], BaseSampler], LogUniformDistribution) -> None

    study = optuna.study.create_study(sampler=sampler_class())
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
def test_discrete_uniform(sampler_class, distribution):
    # type: (typing.Callable[[], BaseSampler], DiscreteUniformDistribution) -> None

    study = optuna.study.create_study(sampler=sampler_class())
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
def test_int(sampler_class, distribution):
    # type: (typing.Callable[[], BaseSampler], IntUniformDistribution) -> None

    study = optuna.study.create_study(sampler=sampler_class())
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
def test_categorical(sampler_class, choices):
    # type: (typing.Callable[[], BaseSampler], Sequence[CategoricalChoiceType]) -> None

    distribution = CategoricalDistribution(choices)

    study = optuna.study.create_study(sampler=sampler_class())

    def sample():
        # type: () -> float

        trial = _create_new_trial(study)
        param_value = study.sampler.sample_independent(study, trial, "x", distribution)
        return distribution.to_internal_repr(param_value)

    points = np.array([sample() for _ in range(100)])

    # 'x' value is corresponding to an index of distribution.choices.
    assert np.all(points >= 0)
    assert np.all(points <= len(distribution.choices) - 1)
    round_points = np.round(points)
    np.testing.assert_almost_equal(round_points, points)


def _create_new_trial(study):
    # type: (Study) -> FrozenTrial

    trial_id = study._storage.create_new_trial(study._study_id)
    return study._storage.get_trial(trial_id)


class FixedSampler(BaseSampler):
    def __init__(self, relative_search_space, relative_params, unknown_param_value):
        # type: (Dict[str, BaseDistribution], Dict[str, Any], Any) -> None

        self.relative_search_space = relative_search_space
        self.relative_params = relative_params
        self.unknown_param_value = unknown_param_value

    def infer_relative_search_space(self, study, trial):
        # type: (Study, FrozenTrial) -> Dict[str, BaseDistribution]

        return self.relative_search_space

    def sample_relative(self, study, trial, search_space):
        # type: (Study, FrozenTrial, Dict[str, BaseDistribution]) -> Dict[str, Any]

        return self.relative_params

    def sample_independent(self, study, trial, param_name, param_distribution):
        # type: (Study, FrozenTrial, str, BaseDistribution) -> Any

        return self.unknown_param_value


def test_sample_relative():
    # type: () -> None

    relative_search_space = {
        "a": UniformDistribution(low=0, high=5),
        "b": CategoricalDistribution(choices=("foo", "bar", "baz")),
        "c": IntUniformDistribution(low=20, high=50),  # Not exist in `relative_params`.
    }  # type: Dict[str, BaseDistribution]
    relative_params = {
        "a": 3.2,
        "b": "baz",
    }
    unknown_param_value = 30

    sampler = FixedSampler(  # type: ignore
        relative_search_space, relative_params, unknown_param_value
    )
    study = optuna.study.create_study(sampler=sampler)

    def objective(trial):
        # type: (Trial) -> float

        # Predefined parameters are sampled by `sample_relative()` method.
        assert trial.suggest_uniform("a", 0, 5) == 3.2
        assert trial.suggest_categorical("b", ["foo", "bar", "baz"]) == "baz"

        # Other parameters are sampled by `sample_independent()` method.
        assert trial.suggest_int("c", 20, 50) == unknown_param_value
        assert trial.suggest_loguniform("d", 1, 100) == unknown_param_value
        assert trial.suggest_uniform("e", 20, 40) == unknown_param_value

        return 0.0

    study.optimize(objective, n_trials=10, catch=())
    for trial in study.trials:
        assert trial.params == {"a": 3.2, "b": "baz", "c": 30, "d": 30, "e": 30}


def test_intersection_search_space() -> None:
    search_space = optuna.samplers.IntersectionSearchSpace()
    study = optuna.create_study()

    # No trial.
    assert search_space.calculate(study) == {}
    assert search_space.calculate(study) == optuna.samplers.intersection_search_space(study)

    # First trial.
    study.optimize(lambda t: t.suggest_uniform("y", -3, 3) + t.suggest_int("x", 0, 10), n_trials=1)
    assert search_space.calculate(study) == {
        "x": IntUniformDistribution(low=0, high=10),
        "y": UniformDistribution(low=-3, high=3),
    }
    assert search_space.calculate(study) == optuna.samplers.intersection_search_space(study)

    # Returning sorted `OrderedDict` instead of `dict`.
    assert search_space.calculate(study, ordered_dict=True) == OrderedDict(
        [
            ("x", IntUniformDistribution(low=0, high=10)),
            ("y", UniformDistribution(low=-3, high=3)),
        ]
    )
    assert search_space.calculate(
        study, ordered_dict=True
    ) == optuna.samplers.intersection_search_space(study, ordered_dict=True)

    # Second trial (only 'y' parameter is suggested in this trial).
    study.optimize(lambda t: t.suggest_uniform("y", -3, 3), n_trials=1)
    assert search_space.calculate(study) == {"y": UniformDistribution(low=-3, high=3)}
    assert search_space.calculate(study) == optuna.samplers.intersection_search_space(study)

    # Failed or pruned trials are not considered in the calculation of
    # an intersection search space.
    def objective(trial, exception):
        # type: (optuna.trial.Trial, Exception) -> float

        trial.suggest_uniform("z", 0, 1)
        raise exception

    study.optimize(lambda t: objective(t, RuntimeError()), n_trials=1, catch=(RuntimeError,))
    study.optimize(lambda t: objective(t, optuna.TrialPruned()), n_trials=1)
    assert search_space.calculate(study) == {"y": UniformDistribution(low=-3, high=3)}
    assert search_space.calculate(study) == optuna.samplers.intersection_search_space(study)

    # If two parameters have the same name but different distributions,
    # those are regarded as different parameters.
    study.optimize(lambda t: t.suggest_uniform("y", -1, 1), n_trials=1)
    assert search_space.calculate(study) == {}
    assert search_space.calculate(study) == optuna.samplers.intersection_search_space(study)

    # The search space remains empty once it is empty.
    study.optimize(lambda t: t.suggest_uniform("y", -3, 3) + t.suggest_int("x", 0, 10), n_trials=1)
    assert search_space.calculate(study) == {}
    assert search_space.calculate(study) == optuna.samplers.intersection_search_space(study)


def test_intersection_search_space_class_with_different_studies() -> None:
    search_space = optuna.samplers.IntersectionSearchSpace()

    with StorageSupplier("sqlite") as storage:
        study0 = optuna.create_study(storage=storage)
        study1 = optuna.create_study(storage=storage)

        search_space.calculate(study0)
        with pytest.raises(ValueError):
            # An `IntersectionSearchSpace` instance isn't supposed to be used for multiple studies.
            search_space.calculate(study1)


@parametrize_sampler
def test_nan_objective_value(sampler_class):
    # type: (typing.Callable[[], BaseSampler]) -> None

    study = optuna.create_study(sampler=sampler_class())

    def objective(trial, base_value):
        # type: (Trial, float) -> float

        return trial.suggest_uniform("x", 0.1, 0.2) + base_value

    # Non NaN objective values.
    for i in range(10, 1, -1):
        study.optimize(lambda t: objective(t, i), n_trials=1, catch=())
    assert int(study.best_value) == 2

    # NaN objective values.
    study.optimize(lambda t: objective(t, float("nan")), n_trials=1, catch=())
    assert int(study.best_value) == 2

    # Non NaN objective value.
    study.optimize(lambda t: objective(t, 1), n_trials=1, catch=())
    assert int(study.best_value) == 1
