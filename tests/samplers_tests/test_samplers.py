from collections import OrderedDict
import pickle
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union
from unittest.mock import patch
import warnings

from _pytest.mark.structures import MarkDecorator
import numpy as np
import pytest

import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.study import Study
from optuna.testing.sampler import DeterministicRelativeSampler
from optuna.trial import FrozenTrial
from optuna.trial import Trial
from optuna.trial import TrialState


parametrize_sampler = pytest.mark.parametrize(
    "sampler_class",
    [
        optuna.samplers.RandomSampler,
        lambda: optuna.samplers.TPESampler(n_startup_trials=0),
        lambda: optuna.samplers.TPESampler(n_startup_trials=0, multivariate=True),
        lambda: optuna.samplers.CmaEsSampler(n_startup_trials=0),
        lambda: optuna.integration.SkoptSampler(
            skopt_kwargs={"base_estimator": "dummy", "n_initial_points": 1}
        ),
        lambda: optuna.integration.PyCmaSampler(n_startup_trials=0),
        optuna.samplers.NSGAIISampler,
    ]
    # TODO(kstoneriv3): Update this after the support for Python 3.6 is stopped.
    + (
        []
        if sys.version_info < (3, 7, 0)
        else [
            lambda: optuna.samplers.QMCSampler(),
        ]
    )
    # TODO(nzw0301): Update this after the support for Python 3.6 is stopped.
    + (
        []
        if sys.version_info < (3, 7, 0)
        else [lambda: optuna.integration.BoTorchSampler(n_startup_trials=0)]
    ),
)
parametrize_relative_sampler = pytest.mark.parametrize(
    "relative_sampler_class",
    [
        lambda: optuna.samplers.TPESampler(n_startup_trials=0, multivariate=True),
        lambda: optuna.samplers.CmaEsSampler(n_startup_trials=0),
        lambda: optuna.integration.SkoptSampler(
            skopt_kwargs={"base_estimator": "dummy", "n_initial_points": 1}
        ),
        lambda: optuna.integration.PyCmaSampler(n_startup_trials=0),
    ],
)
parametrize_multi_objective_sampler = pytest.mark.parametrize(
    "multi_objective_sampler_class",
    [
        optuna.samplers.NSGAIISampler,
        lambda: optuna.samplers.TPESampler(n_startup_trials=0),
    ]
    # TODO(nzw0301): Update this after the support for Python 3.6 is stopped.
    + (
        []
        if sys.version_info < (3, 7, 0)
        else [lambda: optuna.integration.BoTorchSampler(n_startup_trials=0)]
    ),
)


@pytest.mark.parametrize(
    "sampler_class,expected_has_rng,expected_has_another_sampler",
    [
        (optuna.samplers.RandomSampler, True, False),
        (lambda: optuna.samplers.TPESampler(n_startup_trials=0), True, True),
        (lambda: optuna.samplers.TPESampler(n_startup_trials=0, multivariate=True), True, True),
        (lambda: optuna.samplers.CmaEsSampler(n_startup_trials=0), True, True),
        (
            lambda: optuna.integration.SkoptSampler(
                skopt_kwargs={"base_estimator": "dummy", "n_initial_points": 1}
            ),
            False,
            True,
        ),
        (lambda: optuna.integration.PyCmaSampler(n_startup_trials=0), False, True),
        (optuna.samplers.NSGAIISampler, True, True),
        (
            lambda: optuna.samplers.PartialFixedSampler(
                fixed_params={"x": 0}, base_sampler=optuna.samplers.RandomSampler()
            ),
            False,
            True,
        ),
        (lambda: optuna.samplers.GridSampler(search_space={"x": [0]}), True, False),
    ]
    # TODO(kstoneriv3): Update this after the support for Python 3.6 is stopped.
    + (
        []
        if sys.version_info < (3, 7, 0)
        else [
            (lambda: optuna.samplers.QMCSampler(), False, True),
        ]
    )
    # TODO(nzw0301): Remove version constraints if BoTorch supports Python 3.10
    # or Optuna does not support Python 3.6.
    + (
        []
        if sys.version_info >= (3, 10, 0) or sys.version_info < (3, 7, 0)
        else [(lambda: optuna.integration.BoTorchSampler(n_startup_trials=0), False, True)]
    ),
)
def test_sampler_reseed_rng(
    sampler_class: Callable[[], BaseSampler],
    expected_has_rng: bool,
    expected_has_another_sampler: bool,
) -> None:
    def _extract_attr_name_from_sampler_by_cls(sampler: BaseSampler, cls: Any) -> Optional[str]:
        for name, attr in sampler.__dict__.items():
            if isinstance(attr, cls):
                return name
        return None

    sampler = sampler_class()

    rng_name = _extract_attr_name_from_sampler_by_cls(sampler, np.random.RandomState)
    has_rng = rng_name is not None
    assert expected_has_rng == has_rng

    had_sampler_name = _extract_attr_name_from_sampler_by_cls(sampler, BaseSampler)
    has_another_sampler = had_sampler_name is not None
    assert expected_has_another_sampler == has_another_sampler

    if has_rng:
        rng_name = str(rng_name)
        original_random_state = sampler.__dict__[rng_name].get_state()
        sampler.reseed_rng()

        if not isinstance(sampler, optuna.samplers.CmaEsSampler):
            assert str(original_random_state) != str(sampler.__dict__[rng_name].get_state())
        else:
            # CmaEsSampler has a RandomState that is not reseed by its reseed_rng method.
            assert str(original_random_state) == str(sampler.__dict__[rng_name].get_state())

    if has_another_sampler:
        had_sampler_name = str(had_sampler_name)
        had_sampler = sampler.__dict__[had_sampler_name]
        had_sampler_rng_name = _extract_attr_name_from_sampler_by_cls(
            had_sampler, np.random.RandomState
        )

        original_had_sampler_random_state = had_sampler.__dict__[had_sampler_rng_name].get_state()

        with patch.object(
            had_sampler,
            "reseed_rng",
            wraps=had_sampler.reseed_rng,
        ) as mock_object:
            sampler.reseed_rng()
            assert mock_object.call_count == 1

        had_sampler = sampler.__dict__[had_sampler_name]
        assert str(original_had_sampler_random_state) != str(
            had_sampler.__dict__[had_sampler_rng_name].get_state()
        )


def parametrize_suggest_method(name: str) -> MarkDecorator:
    return pytest.mark.parametrize(
        f"suggest_method_{name}",
        [
            lambda t: t.suggest_float(name, 0, 10),
            lambda t: t.suggest_int(name, 0, 10),
            lambda t: cast(float, t.suggest_categorical(name, [0, 1, 2])),
            lambda t: t.suggest_float(name, 0, 10, step=0.5),
            lambda t: t.suggest_float(name, 1e-7, 10, log=True),
            lambda t: t.suggest_int(name, 1, 10, log=True),
        ],
    )


@pytest.mark.parametrize(
    "sampler_class",
    [
        lambda: optuna.samplers.CmaEsSampler(n_startup_trials=0),
        lambda: optuna.integration.SkoptSampler(
            skopt_kwargs={"base_estimator": "dummy", "n_initial_points": 1}
        ),
        lambda: optuna.integration.PyCmaSampler(n_startup_trials=0),
    ],
)
def test_raise_error_for_samplers_during_multi_objectives(
    sampler_class: Callable[[], BaseSampler]
) -> None:

    study = optuna.study.create_study(directions=["maximize", "maximize"], sampler=sampler_class())

    distribution = FloatDistribution(0.0, 1.0)
    with pytest.raises(ValueError):
        study.sampler.sample_independent(study, _create_new_trial(study), "x", distribution)

    with pytest.raises(ValueError):
        trial = _create_new_trial(study)
        study.sampler.sample_relative(
            study, trial, study.sampler.infer_relative_search_space(study, trial)
        )


@pytest.mark.parametrize("seed", [None, 0, 169208])
def test_pickle_random_sampler(seed: Optional[int]) -> None:

    sampler = optuna.samplers.RandomSampler(seed)
    restored_sampler = pickle.loads(pickle.dumps(sampler))
    assert sampler._rng.bytes(10) == restored_sampler._rng.bytes(10)


@parametrize_sampler
@pytest.mark.parametrize(
    "distribution",
    [
        FloatDistribution(-1.0, 1.0),
        FloatDistribution(0.0, 1.0),
        FloatDistribution(-1.0, 0.0),
        FloatDistribution(1e-7, 1.0, log=True),
        FloatDistribution(-10, 10, step=0.1),
        FloatDistribution(-10.2, 10.2, step=0.1),
    ],
)
def test_float(
    sampler_class: Callable[[], BaseSampler],
    distribution: FloatDistribution,
) -> None:

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

    if distribution.step is not None:
        # Check all points are multiples of distribution.step.
        points -= distribution.low
        points /= distribution.step
        round_points = np.round(points)
        np.testing.assert_almost_equal(round_points, points)


@parametrize_sampler
@pytest.mark.parametrize(
    "distribution",
    [
        IntDistribution(-10, 10),
        IntDistribution(0, 10),
        IntDistribution(-10, 0),
        IntDistribution(-10, 10, step=2),
        IntDistribution(0, 10, step=2),
        IntDistribution(-10, 0, step=2),
        IntDistribution(1, 100, log=True),
    ],
)
def test_int(sampler_class: Callable[[], BaseSampler], distribution: IntDistribution) -> None:

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
def test_categorical(
    sampler_class: Callable[[], BaseSampler], choices: Sequence[CategoricalChoiceType]
) -> None:

    distribution = CategoricalDistribution(choices)

    study = optuna.study.create_study(sampler=sampler_class())

    def sample() -> float:

        trial = _create_new_trial(study)
        param_value = study.sampler.sample_independent(study, trial, "x", distribution)
        return float(distribution.to_internal_repr(param_value))

    points = np.asarray([sample() for i in range(100)])

    # 'x' value is corresponding to an index of distribution.choices.
    assert np.all(points >= 0)
    assert np.all(points <= len(distribution.choices) - 1)
    round_points = np.round(points)
    np.testing.assert_almost_equal(round_points, points)


@parametrize_relative_sampler
@pytest.mark.parametrize(
    "x_distribution",
    [
        FloatDistribution(-1.0, 1.0),
        FloatDistribution(1e-7, 1.0, log=True),
        FloatDistribution(-10, 10, step=0.5),
        IntDistribution(3, 10),
        IntDistribution(1, 100, log=True),
        IntDistribution(3, 10, step=2),
    ],
)
@pytest.mark.parametrize(
    "y_distribution",
    [
        FloatDistribution(-1.0, 1.0),
        FloatDistribution(1e-7, 1.0, log=True),
        FloatDistribution(-10, 10, step=0.5),
        IntDistribution(3, 10),
        IntDistribution(1, 100, log=True),
        IntDistribution(3, 10, step=2),
    ],
)
def test_sample_relative_numerical(
    relative_sampler_class: Callable[[], BaseSampler],
    x_distribution: BaseDistribution,
    y_distribution: BaseDistribution,
) -> None:

    search_space: Dict[str, BaseDistribution] = OrderedDict(x=x_distribution, y=y_distribution)
    study = optuna.study.create_study(sampler=relative_sampler_class())
    trial = study.ask(search_space)
    study.tell(trial, sum(trial.params.values()))

    def sample() -> List[Union[int, float]]:
        params = study.sampler.sample_relative(study, _create_new_trial(study), search_space)
        return [params[name] for name in search_space]

    points = np.array([sample() for _ in range(10)])
    for i, distribution in enumerate(search_space.values()):
        assert isinstance(
            distribution,
            (
                FloatDistribution,
                IntDistribution,
            ),
        )
        assert np.all(points[:, i] >= distribution.low)
        assert np.all(points[:, i] <= distribution.high)
    for param_value, distribution in zip(sample(), search_space.values()):
        assert not isinstance(param_value, np.floating)
        assert not isinstance(param_value, np.integer)
        if isinstance(distribution, IntDistribution):
            assert isinstance(param_value, int)
        else:
            assert isinstance(param_value, float)


@parametrize_relative_sampler
def test_sample_relative_categorical(relative_sampler_class: Callable[[], BaseSampler]) -> None:

    search_space: Dict[str, BaseDistribution] = OrderedDict(
        x=CategoricalDistribution([1, 10, 100]), y=CategoricalDistribution([-1, -10, -100])
    )
    study = optuna.study.create_study(sampler=relative_sampler_class())
    trial = study.ask(search_space)
    study.tell(trial, sum(trial.params.values()))

    def sample() -> List[float]:
        params = study.sampler.sample_relative(study, _create_new_trial(study), search_space)
        return [params[name] for name in search_space]

    points = np.array([sample() for _ in range(10)])
    for i, distribution in enumerate(search_space.values()):
        assert isinstance(distribution, CategoricalDistribution)
        assert np.all([v in distribution.choices for v in points[:, i]])
    for param_value in sample():
        assert not isinstance(param_value, np.floating)
        assert not isinstance(param_value, np.integer)
        assert isinstance(param_value, int)


@parametrize_relative_sampler
@pytest.mark.parametrize(
    "x_distribution",
    [
        FloatDistribution(-1.0, 1.0),
        FloatDistribution(1e-7, 1.0, log=True),
        FloatDistribution(-10, 10, step=0.5),
        IntDistribution(1, 10),
        IntDistribution(1, 100, log=True),
    ],
)
def test_sample_relative_mixed(
    relative_sampler_class: Callable[[], BaseSampler], x_distribution: BaseDistribution
) -> None:

    search_space: Dict[str, BaseDistribution] = OrderedDict(
        x=x_distribution, y=CategoricalDistribution([-1, -10, -100])
    )
    study = optuna.study.create_study(sampler=relative_sampler_class())
    trial = study.ask(search_space)
    study.tell(trial, sum(trial.params.values()))

    def sample() -> List[float]:
        params = study.sampler.sample_relative(study, _create_new_trial(study), search_space)
        return [params[name] for name in search_space]

    points = np.array([sample() for _ in range(10)])
    assert isinstance(
        search_space["x"],
        (
            FloatDistribution,
            IntDistribution,
        ),
    )
    assert np.all(points[:, 0] >= search_space["x"].low)
    assert np.all(points[:, 0] <= search_space["x"].high)
    assert isinstance(search_space["y"], CategoricalDistribution)
    assert np.all([v in search_space["y"].choices for v in points[:, 1]])
    for param_value, distribution in zip(sample(), search_space.values()):
        assert not isinstance(param_value, np.floating)
        assert not isinstance(param_value, np.integer)
        if isinstance(
            distribution,
            (
                IntDistribution,
                CategoricalDistribution,
            ),
        ):
            assert isinstance(param_value, int)
        else:
            assert isinstance(param_value, float)


@parametrize_sampler
def test_conditional_sample_independent(sampler_class: Callable[[], BaseSampler]) -> None:
    # This test case reproduces the error reported in #2734.
    # See https://github.com/optuna/optuna/pull/2734#issuecomment-857649769.

    study = optuna.study.create_study(sampler=sampler_class())
    categorical_distribution = CategoricalDistribution(choices=["x", "y"])
    dependent_distribution = CategoricalDistribution(choices=["a", "b"])

    study.add_trial(
        optuna.create_trial(
            params={"category": "x", "x": "a"},
            distributions={"category": categorical_distribution, "x": dependent_distribution},
            value=0.1,
        )
    )

    study.add_trial(
        optuna.create_trial(
            params={"category": "y", "y": "b"},
            distributions={"category": categorical_distribution, "y": dependent_distribution},
            value=0.1,
        )
    )

    _trial = _create_new_trial(study)
    category = study.sampler.sample_independent(
        study, _trial, "category", categorical_distribution
    )
    assert category in ["x", "y"]
    value = study.sampler.sample_independent(study, _trial, category, dependent_distribution)
    assert value in ["a", "b"]


def _create_new_trial(study: Study) -> FrozenTrial:

    trial_id = study._storage.create_new_trial(study._study_id)
    return study._storage.get_trial(trial_id)


class FixedSampler(BaseSampler):
    def __init__(
        self,
        relative_search_space: Dict[str, BaseDistribution],
        relative_params: Dict[str, Any],
        unknown_param_value: Any,
    ) -> None:

        self.relative_search_space = relative_search_space
        self.relative_params = relative_params
        self.unknown_param_value = unknown_param_value

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:

        return self.relative_search_space

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:

        return self.relative_params

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:

        return self.unknown_param_value


def test_sample_relative() -> None:

    relative_search_space: Dict[str, BaseDistribution] = {
        "a": FloatDistribution(low=0, high=5),
        "b": CategoricalDistribution(choices=("foo", "bar", "baz")),
        "c": IntDistribution(low=20, high=50),  # Not exist in `relative_params`.
    }
    relative_params = {
        "a": 3.2,
        "b": "baz",
    }
    unknown_param_value = 30

    sampler = FixedSampler(  # type: ignore
        relative_search_space, relative_params, unknown_param_value
    )
    study = optuna.study.create_study(sampler=sampler)

    def objective(trial: Trial) -> float:

        # Predefined parameters are sampled by `sample_relative()` method.
        assert trial.suggest_float("a", 0, 5) == 3.2
        assert trial.suggest_categorical("b", ["foo", "bar", "baz"]) == "baz"

        # Other parameters are sampled by `sample_independent()` method.
        assert trial.suggest_int("c", 20, 50) == unknown_param_value
        assert trial.suggest_float("d", 1, 100, log=True) == unknown_param_value
        assert trial.suggest_float("e", 20, 40) == unknown_param_value

        return 0.0

    study.optimize(objective, n_trials=10, catch=())
    for trial in study.trials:
        assert trial.params == {"a": 3.2, "b": "baz", "c": 30, "d": 30, "e": 30}


@parametrize_sampler
def test_nan_objective_value(sampler_class: Callable[[], BaseSampler]) -> None:

    study = optuna.create_study(sampler=sampler_class())

    def objective(trial: Trial, base_value: float) -> float:

        return trial.suggest_float("x", 0.1, 0.2) + base_value

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


@parametrize_sampler
def test_partial_fixed_sampling(sampler_class: Callable[[], BaseSampler]) -> None:

    study = optuna.create_study(sampler=sampler_class())

    def objective(trial: Trial) -> float:
        x = trial.suggest_float("x", -1, 1)
        y = trial.suggest_int("y", -1, 1)
        z = trial.suggest_float("z", -1, 1)
        return x + y + z

    # First trial.
    study.optimize(objective, n_trials=1)

    # Second trial. Here, the parameter ``y`` is fixed as 0.
    fixed_params = {"y": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        study.sampler = optuna.samplers.PartialFixedSampler(fixed_params, study.sampler)
    study.optimize(objective, n_trials=1)
    trial_params = study.trials[-1].params
    assert trial_params["y"] == fixed_params["y"]


@parametrize_multi_objective_sampler
@pytest.mark.parametrize(
    "distribution",
    [
        FloatDistribution(-1.0, 1.0),
        FloatDistribution(0.0, 1.0),
        FloatDistribution(-1.0, 0.0),
        FloatDistribution(1e-7, 1.0, log=True),
        FloatDistribution(-10, 10, step=0.1),
        FloatDistribution(-10.2, 10.2, step=0.1),
        IntDistribution(-10, 10),
        IntDistribution(0, 10),
        IntDistribution(-10, 0),
        IntDistribution(-10, 10, step=2),
        IntDistribution(0, 10, step=2),
        IntDistribution(-10, 0, step=2),
        IntDistribution(1, 100, log=True),
        CategoricalDistribution((1, 2, 3)),
        CategoricalDistribution(("a", "b", "c")),
        CategoricalDistribution((1, "a")),
    ],
)
def test_multi_objective_sample_independent(
    multi_objective_sampler_class: Callable[[], BaseSampler], distribution: BaseDistribution
) -> None:
    study = optuna.study.create_study(
        directions=["minimize", "maximize"], sampler=multi_objective_sampler_class()
    )
    for i in range(100):
        value = study.sampler.sample_independent(
            study, _create_new_trial(study), "x", distribution
        )
        assert distribution._contains(distribution.to_internal_repr(value))

        if not isinstance(distribution, CategoricalDistribution):
            # Please see https://github.com/optuna/optuna/pull/393 why this assertion is needed.
            assert not isinstance(value, np.floating)

        if isinstance(distribution, FloatDistribution):
            if distribution.step is not None:
                # Check the value is a multiple of `distribution.step` which is
                # the quantization interval of the distribution.
                value -= distribution.low
                value /= distribution.step
                round_value = np.round(value)
                np.testing.assert_almost_equal(round_value, value)


def test_after_trial() -> None:
    n_calls = 0
    n_trials = 3

    class SamplerAfterTrial(DeterministicRelativeSampler):
        def after_trial(
            self,
            study: Study,
            trial: FrozenTrial,
            state: TrialState,
            values: Optional[Sequence[float]],
        ) -> None:
            assert len(study.trials) - 1 == trial.number
            assert trial.state == TrialState.RUNNING
            assert trial.values is None
            assert state == TrialState.COMPLETE
            assert values is not None
            assert len(values) == 2
            nonlocal n_calls
            n_calls += 1

    sampler = SamplerAfterTrial({}, {})
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)

    study.optimize(lambda t: [t.suggest_float("y", -3, 3), t.suggest_int("x", 0, 10)], n_trials=3)

    assert n_calls == n_trials


def test_after_trial_pruning() -> None:
    n_calls = 0
    n_trials = 3

    class SamplerAfterTrial(DeterministicRelativeSampler):
        def after_trial(
            self,
            study: Study,
            trial: FrozenTrial,
            state: TrialState,
            values: Optional[Sequence[float]],
        ) -> None:
            assert len(study.trials) - 1 == trial.number
            assert trial.state == TrialState.RUNNING
            assert trial.values is None
            assert state == TrialState.PRUNED
            assert values is None
            nonlocal n_calls
            n_calls += 1

    sampler = SamplerAfterTrial({}, {})
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)

    def objective(trial: Trial) -> Any:
        raise optuna.TrialPruned

    study.optimize(objective, n_trials=n_trials)

    assert n_calls == n_trials


def test_after_trial_failing() -> None:
    n_calls = 0
    n_trials = 3

    class SamplerAfterTrial(DeterministicRelativeSampler):
        def after_trial(
            self,
            study: Study,
            trial: FrozenTrial,
            state: TrialState,
            values: Optional[Sequence[float]],
        ) -> None:
            assert len(study.trials) - 1 == trial.number
            assert trial.state == TrialState.RUNNING
            assert trial.values is None
            assert state == TrialState.FAIL
            assert values is None
            nonlocal n_calls
            n_calls += 1

    sampler = SamplerAfterTrial({}, {})
    study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)

    def objective(trial: Trial) -> Any:
        raise NotImplementedError  # Arbitrary error for testing purpose.

    with pytest.raises(NotImplementedError):
        study.optimize(objective, n_trials=n_trials)

    # Called once after the first failing trial before returning from optimize.
    assert n_calls == 1


def test_after_trial_failing_in_after_trial() -> None:
    n_calls = 0
    n_trials = 3

    class SamplerAfterTrialAlwaysFail(DeterministicRelativeSampler):
        def after_trial(
            self,
            study: Study,
            trial: FrozenTrial,
            state: TrialState,
            values: Optional[Sequence[float]],
        ) -> None:
            nonlocal n_calls
            n_calls += 1
            raise NotImplementedError  # Arbitrary error for testing purpose.

    sampler = SamplerAfterTrialAlwaysFail({}, {})
    study = optuna.create_study(sampler=sampler)

    with pytest.raises(NotImplementedError):
        study.optimize(lambda t: t.suggest_int("x", 0, 10), n_trials=n_trials)

    assert len(study.trials) == 1
    assert n_calls == 1

    sampler = SamplerAfterTrialAlwaysFail({}, {})
    study = optuna.create_study(sampler=sampler)

    # Not affected by `catch`.
    with pytest.raises(NotImplementedError):
        study.optimize(
            lambda t: t.suggest_int("x", 0, 10), n_trials=n_trials, catch=(NotImplementedError,)
        )

    assert len(study.trials) == 1
    assert n_calls == 2


def test_after_trial_with_study_tell() -> None:
    n_calls = 0

    class SamplerAfterTrial(DeterministicRelativeSampler):
        def after_trial(
            self,
            study: Study,
            trial: FrozenTrial,
            state: TrialState,
            values: Optional[Sequence[float]],
        ) -> None:
            nonlocal n_calls
            n_calls += 1

    sampler = SamplerAfterTrial({}, {})
    study = optuna.create_study(sampler=sampler)

    assert n_calls == 0

    study.tell(study.ask(), 1.0)

    assert n_calls == 1


@parametrize_sampler
def test_sample_single_distribution(sampler_class: Callable[[], BaseSampler]) -> None:

    relative_search_space = {
        "a": CategoricalDistribution([1]),
        "b": IntDistribution(low=1, high=1),
        "c": IntDistribution(low=1, high=1, log=True),
        "d": FloatDistribution(low=1.0, high=1.0),
        "e": FloatDistribution(low=1.0, high=1.0, log=True),
        "f": FloatDistribution(low=1.0, high=1.0, step=1.0),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = sampler_class()
    study = optuna.study.create_study(sampler=sampler)

    # We need to test the construction of the model, so we should set `n_trials >= 2`.
    for _ in range(2):
        trial = study.ask(fixed_distributions=relative_search_space)
        study.tell(trial, 1.0)
        for param_name in relative_search_space.keys():
            assert trial.params[param_name] == 1


@parametrize_sampler
@parametrize_suggest_method("x")
def test_single_parameter_objective(
    sampler_class: Callable[[], BaseSampler], suggest_method_x: Callable[[Trial], float]
) -> None:
    def objective(trial: Trial) -> float:
        return suggest_method_x(trial)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = sampler_class()

    study = optuna.study.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 10
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


@parametrize_sampler
def test_conditional_parameter_objective(sampler_class: Callable[[], BaseSampler]) -> None:
    def objective(trial: Trial) -> float:
        x = trial.suggest_categorical("x", [True, False])
        if x:
            return trial.suggest_float("y", 0, 1)
        return trial.suggest_float("z", 0, 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = sampler_class()

    study = optuna.study.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 10
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


@parametrize_sampler
@parametrize_suggest_method("x")
@parametrize_suggest_method("y")
def test_combination_of_different_distributions_objective(
    sampler_class: Callable[[], BaseSampler],
    suggest_method_x: Callable[[Trial], float],
    suggest_method_y: Callable[[Trial], float],
) -> None:
    def objective(trial: Trial) -> float:
        return suggest_method_x(trial) + suggest_method_y(trial)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = sampler_class()

    study = optuna.study.create_study(sampler=sampler)
    study.optimize(objective, n_trials=10)

    assert len(study.trials) == 10
    assert all(t.state == TrialState.COMPLETE for t in study.trials)


@parametrize_sampler
@pytest.mark.parametrize(
    "second_low,second_high",
    [
        (0, 5),  # Narrow range.
        (0, 20),  # Expand range.
        (20, 30),  # Set non-overlapping range.
    ],
)
def test_dynamic_range_objective(
    sampler_class: Callable[[], BaseSampler], second_low: int, second_high: int
) -> None:
    def objective(trial: Trial, low: int, high: int) -> float:
        v = trial.suggest_float("x", low, high)
        v += trial.suggest_int("y", low, high)
        return v

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = sampler_class()

    study = optuna.study.create_study(sampler=sampler)
    study.optimize(lambda t: objective(t, 0, 10), n_trials=10)
    study.optimize(lambda t: objective(t, second_low, second_high), n_trials=10)

    assert len(study.trials) == 20
    assert all(t.state == TrialState.COMPLETE for t in study.trials)
