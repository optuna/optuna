from collections import Counter
from collections import defaultdict
import copy
import itertools
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from unittest.mock import Mock
from unittest.mock import patch
import warnings

import numpy as np
import pytest

import optuna
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.samplers import NSGAIIISampler
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers.nsgaii import BaseCrossover
from optuna.samplers.nsgaii import BLXAlphaCrossover
from optuna.samplers.nsgaii import SBXCrossover
from optuna.samplers.nsgaii import SPXCrossover
from optuna.samplers.nsgaii import UNDXCrossover
from optuna.samplers.nsgaii import UniformCrossover
from optuna.samplers.nsgaii import VSBXCrossover
from optuna.samplers.nsgaii._crossover import _inlined_categorical_uniform_crossover
from optuna.samplers.nsgaii._sampler import _constrained_dominates
from optuna.samplers.nsgaiii import _associate
from optuna.samplers.nsgaiii import _niching
from optuna.samplers.nsgaiii import _POPULATION_CACHE_KEY_PREFIX
from optuna.samplers.nsgaiii import generate_default_reference_point
from optuna.study._study_direction import StudyDirection
from optuna.trial import create_trial
from optuna.trial import FrozenTrial


def _nan_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True

    return a == b


def test_population_size() -> None:
    reference_points = np.array([[1.0]])

    # Set `population_size` to 10.
    sampler = NSGAIIISampler(reference_points, population_size=10)

    study = optuna.create_study(directions=["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float("x", 0, 9)], n_trials=40)

    generations = Counter(
        [t.system_attrs[optuna.samplers.nsgaiii._GENERATION_KEY] for t in study.trials]
    )
    assert generations == {0: 10, 1: 10, 2: 10, 3: 10}

    # Set `population_size` to 2.
    sampler = NSGAIIISampler(reference_points, population_size=2)

    study = optuna.create_study(directions=["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float("x", 0, 9)], n_trials=40)

    generations = Counter(
        [t.system_attrs[optuna.samplers.nsgaiii._GENERATION_KEY] for t in study.trials]
    )
    assert generations == {i: 2 for i in range(20)}

    # Invalid population size.
    with pytest.raises(ValueError):
        # Less than 2.
        NSGAIIISampler(reference_points, population_size=1)

    with pytest.raises(TypeError):
        # Not an integer.
        NSGAIIISampler(reference_points, population_size=2.5)  # type: ignore[arg-type]


def test_mutation_prob() -> None:
    reference_points = np.array([[1.0]])

    NSGAIIISampler(reference_points, mutation_prob=None)
    NSGAIIISampler(reference_points, mutation_prob=0.0)
    NSGAIIISampler(reference_points, mutation_prob=0.5)
    NSGAIIISampler(reference_points, mutation_prob=1.0)

    with pytest.raises(ValueError):
        NSGAIIISampler(reference_points, mutation_prob=-0.5)

    with pytest.raises(ValueError):
        NSGAIIISampler(reference_points, mutation_prob=1.1)


def test_crossover_prob() -> None:
    reference_points = np.array([[1.0]])

    NSGAIIISampler(reference_points, crossover_prob=0.0)
    NSGAIIISampler(reference_points, crossover_prob=0.5)
    NSGAIIISampler(reference_points, crossover_prob=1.0)

    with pytest.raises(ValueError):
        NSGAIIISampler(reference_points, crossover_prob=-0.5)

    with pytest.raises(ValueError):
        NSGAIIISampler(reference_points, crossover_prob=1.1)


def test_swapping_prob() -> None:
    reference_points = np.array([[1.0]])

    NSGAIIISampler(reference_points, swapping_prob=0.0)
    NSGAIIISampler(reference_points, swapping_prob=0.5)
    NSGAIIISampler(reference_points, swapping_prob=1.0)

    with pytest.raises(ValueError):
        NSGAIIISampler(reference_points, swapping_prob=-0.5)

    with pytest.raises(ValueError):
        NSGAIIISampler(reference_points, swapping_prob=1.1)


def test_constraints_func_none() -> None:
    n_trials = 4
    n_objectives = 2

    reference_points = np.array([[1.0, 0.0], [0.0, 1.0]])
    sampler = NSGAIIISampler(reference_points, population_size=2)

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler)
    study.optimize(
        lambda t: [t.suggest_float(f"x{i}", 0, 1) for i in range(n_objectives)],
        n_trials=n_trials,
    )

    assert len(study.trials) == n_trials
    for trial in study.trials:
        assert _CONSTRAINTS_KEY not in trial.system_attrs


@pytest.mark.parametrize("constraint_value", [-1.0, 0.0, 1.0, -float("inf"), float("inf")])
def test_constraints_func(constraint_value: float) -> None:
    n_trials = 4
    n_objectives = 2
    constraints_func_call_count = 0

    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        nonlocal constraints_func_call_count
        constraints_func_call_count += 1

        return (constraint_value + trial.number,)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        reference_points = np.array([[1.0, 0.0], [0.0, 1.0]])
        sampler = NSGAIIISampler(
            reference_points, population_size=2, constraints_func=constraints_func
        )

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler)
    study.optimize(
        lambda t: [t.suggest_float(f"x{i}", 0, 1) for i in range(n_objectives)],
        n_trials=n_trials,
    )

    assert len(study.trials) == n_trials
    assert constraints_func_call_count == n_trials
    for trial in study.trials:
        for x, y in zip(trial.system_attrs[_CONSTRAINTS_KEY], (constraint_value + trial.number,)):
            assert x == y


def test_constraints_func_nan() -> None:
    n_trials = 4
    n_objectives = 2

    def constraints_func(_: FrozenTrial) -> Sequence[float]:
        return (float("nan"),)

    with warnings.catch_warnings():
        reference_points = np.array([[1.0, 0.0], [0.0, 1.0]])
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = NSGAIIISampler(
            reference_points, population_size=2, constraints_func=constraints_func
        )

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler)
    with pytest.raises(ValueError):
        study.optimize(
            lambda t: [t.suggest_float(f"x{i}", 0, 1) for i in range(n_objectives)],
            n_trials=n_trials,
        )

    trials = study.get_trials()
    assert len(trials) == 1  # The error stops optimization, but completed trials are recorded.
    assert all(0 <= x <= 1 for x in trials[0].params.values())  # The params are normal.
    assert trials[0].values == list(trials[0].params.values())  # The values are normal.
    assert trials[0].system_attrs[_CONSTRAINTS_KEY] is None  # None is set for constraints.


# TODO(Shinichi) Remove after separating _fast_non_dominated_sort() from NSGA typed samplers
def _assert_population_per_rank(
    trials: Sequence[FrozenTrial],
    direction: Sequence[StudyDirection],
    population_per_rank: Sequence[Sequence[FrozenTrial]],
) -> None:
    # Check that the number of trials do not change.
    flattened = [trial for rank in population_per_rank for trial in rank]
    assert len(flattened) == len(trials)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # Check that the trials in the same rank do not dominate each other.
        for i in range(len(population_per_rank)):
            for trial1 in population_per_rank[i]:
                for trial2 in population_per_rank[i]:
                    assert not _constrained_dominates(trial1, trial2, direction)

        # Check that each trial is dominated by some trial in the rank above.
        for i in range(len(population_per_rank) - 1):
            for trial2 in population_per_rank[i + 1]:
                assert any(
                    _constrained_dominates(trial1, trial2, direction)
                    for trial1 in population_per_rank[i]
                )


# TODO(Shinichi) Remove after separating _fast_non_dominated_sort() from NSGA typed samplers
@pytest.mark.parametrize("direction1", [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
@pytest.mark.parametrize("direction2", [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
def test_fast_non_dominated_sort_no_constraints(
    direction1: StudyDirection, direction2: StudyDirection
) -> None:
    reference_points = np.array([[1.0]])
    sampler = NSGAIIISampler(reference_points)

    directions = [direction1, direction2]
    value_list = [10, 20, 20, 30, float("inf"), float("inf"), -float("inf")]
    values = [[v1, v2] for v1 in value_list for v2 in value_list]

    trials = [_create_frozen_trial(i, v) for i, v in enumerate(values)]
    population_per_rank = sampler._fast_non_dominated_sort(copy.copy(trials), directions)
    _assert_population_per_rank(trials, directions, population_per_rank)


# TODO(Shinichi) Remove after separating _fast_non_dominated_sort() from NSGA typed samplers
def test_fast_non_dominated_sort_with_constraints() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        reference_points = np.array([[1.0, 0.0], [0.0, 1.0]])
        sampler = NSGAIIISampler(reference_points, constraints_func=lambda _: [0])

    value_list = [10, 20, 20, 30, float("inf"), float("inf"), -float("inf")]
    values = [[v1, v2] for v1 in value_list for v2 in value_list]

    constraint_list = [-float("inf"), -2, 0, 1, 2, 3, float("inf")]
    constraints = [[c1, c2] for c1 in constraint_list for c2 in constraint_list]

    trials = [
        _create_frozen_trial(i, v, c)
        for i, (v, c) in enumerate(itertools.product(values, constraints))
    ]
    directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]
    population_per_rank = sampler._fast_non_dominated_sort(copy.copy(trials), directions)
    _assert_population_per_rank(trials, directions, population_per_rank)


# TODO(Shinichi) Remove after separating _fast_non_dominated_sort() from NSGA typed samplers
def test_fast_non_dominated_sort_with_nan_constraint() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        reference_points = np.array([[1.0, 0.0], [0.0, 1.0]])
        sampler = NSGAIIISampler(reference_points, constraints_func=lambda _: [0])
    directions = [StudyDirection.MINIMIZE, StudyDirection.MINIMIZE]
    with pytest.raises(ValueError):
        sampler._fast_non_dominated_sort(
            [_create_frozen_trial(0, [1], [0, float("nan")])], directions
        )


# TODO(Shinichi) Remove after separating _fast_non_dominated_sort() from NSGA typed samplers
@pytest.mark.parametrize(
    "values_and_constraints",
    [
        [([10], None), ([20], None), ([20], [0]), ([20], [1]), ([30], [-1])],
        [
            ([50, 30], None),
            ([30, 50], None),
            ([20, 20], [3, 3]),
            ([30, 10], [0, -1]),
            ([15, 15], [4, 4]),
        ],
    ],
)
def test_fast_non_dominated_sort_missing_constraint_values(
    values_and_constraints: Sequence[Tuple[Sequence[float], Sequence[float]]]
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        reference_points = np.array([[1.0, 0.0], [0.0, 1.0]])
        sampler = NSGAIIISampler(reference_points, constraints_func=lambda _: [0])

    values_dim = len(values_and_constraints[0][0])
    for directions in itertools.product(
        [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE], repeat=values_dim
    ):
        trials = [_create_frozen_trial(i, v, c) for i, (v, c) in enumerate(values_and_constraints)]

        with pytest.warns(UserWarning):
            population_per_rank = sampler._fast_non_dominated_sort(
                copy.copy(trials), list(directions)
            )
        _assert_population_per_rank(trials, list(directions), population_per_rank)


# TODO(Shinichi) Remove after separating _fast_non_dominated_sort() from NSGA typed samplers
@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_fast_non_dominated_sort_empty(n_dims: int) -> None:
    for directions in itertools.product(
        [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE], repeat=n_dims
    ):
        trials: List[FrozenTrial] = []
        reference_points = np.array([[1.0, 0.0], [0.0, 1.0]])
        sampler = NSGAIIISampler(reference_points)
        population_per_rank = sampler._fast_non_dominated_sort(trials, list(directions))
        assert population_per_rank == []


def test_study_system_attr_for_population_cache() -> None:
    reference_points = np.array([[1.0]])
    sampler = NSGAIIISampler(reference_points, population_size=10)
    study = optuna.create_study(directions=["minimize"], sampler=sampler)

    def get_cached_entries(
        study: optuna.study.Study,
    ) -> List[Tuple[int, List[int]]]:
        study_system_attrs = study._storage.get_study_system_attrs(study._study_id)
        return [
            v for k, v in study_system_attrs.items() if k.startswith(_POPULATION_CACHE_KEY_PREFIX)
        ]

    study.optimize(lambda t: [t.suggest_float("x", 0, 9)], n_trials=10)
    cached_entries = get_cached_entries(study)
    assert len(cached_entries) == 0

    study.optimize(lambda t: [t.suggest_float("x", 0, 9)], n_trials=1)
    cached_entries = get_cached_entries(study)
    assert len(cached_entries) == 1
    assert cached_entries[0][0] == 0  # Cached generation.
    assert len(cached_entries[0][1]) == 10  # Population size.

    study.optimize(lambda t: [t.suggest_float("x", 0, 9)], n_trials=10)
    cached_entries = get_cached_entries(study)
    assert len(cached_entries) == 1
    assert cached_entries[0][0] == 1  # Cached generation.
    assert len(cached_entries[0][1]) == 10  # Population size.


def test_constraints_func_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        reference_points = np.array([[1.0]])
        NSGAIIISampler(reference_points, constraints_func=lambda _: [0])


# TODO(ohta): Consider to move this utility function to `optuna.testing` module.
def _create_frozen_trial(
    number: int, values: Sequence[float], constraints: Optional[Sequence[float]] = None
) -> optuna.trial.FrozenTrial:
    trial = optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        values=list(values),
        system_attrs={} if constraints is None else {_CONSTRAINTS_KEY: list(constraints)},
    )
    trial.number = number
    trial._trial_id = number
    return trial


def test_call_after_trial_of_random_sampler() -> None:
    reference_points = np.array([[1.0]])
    sampler = NSGAIIISampler(reference_points)
    study = optuna.create_study(sampler=sampler)
    with patch.object(
        sampler._random_sampler, "after_trial", wraps=sampler._random_sampler.after_trial
    ) as mock_object:
        study.optimize(lambda _: 1.0, n_trials=1)
        assert mock_object.call_count == 1


parametrize_crossover_population = pytest.mark.parametrize(
    "crossover,population_size",
    [
        (UniformCrossover(), 2),
        (BLXAlphaCrossover(), 2),
        (SBXCrossover(), 2),
        (VSBXCrossover(), 2),
        (UNDXCrossover(), 2),
        (UNDXCrossover(), 3),
    ],
)


@parametrize_crossover_population
@pytest.mark.parametrize("n_objectives", [1, 2, 3])
def test_crossover_objectives(
    n_objectives: int, crossover: BaseSampler, population_size: int
) -> None:
    n_trials = 8
    reference_points = generate_default_reference_point(n_objectives)
    sampler = NSGAIIISampler(reference_points, population_size=population_size)

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler)
    study.optimize(
        lambda t: [t.suggest_float(f"x{i}", 0, 1) for i in range(n_objectives)],
        n_trials=n_trials,
    )
    print()

    assert len(study.trials) == n_trials


@parametrize_crossover_population
@pytest.mark.parametrize("n_params", [1, 2, 3])
def test_crossover_dims(n_params: int, crossover: BaseSampler, population_size: int) -> None:
    def objective(trial: optuna.Trial) -> float:
        xs = [trial.suggest_float(f"x{dim}", -10, 10) for dim in range(n_params)]
        return sum(xs)

    n_objectives = 1
    n_trials = 8

    reference_points = np.array([[0.0] * n_objectives])
    reference_points[0, 0] = 1.0
    sampler = NSGAIIISampler(reference_points, population_size=population_size)

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    assert len(study.trials) == n_trials


@pytest.mark.parametrize(
    "crossover,population_size",
    [
        (UniformCrossover(), 1),
        (BLXAlphaCrossover(), 1),
        (SBXCrossover(), 1),
        (VSBXCrossover(), 1),
        (UNDXCrossover(), 2),
        (SPXCrossover(), 2),
    ],
)
def test_crossover_invalid_population(crossover: BaseCrossover, population_size: int) -> None:
    with pytest.raises(ValueError):
        reference_points = np.array([[1.0]])
        NSGAIIISampler(reference_points, population_size=population_size, crossover=crossover)


@pytest.mark.parametrize(
    "crossover",
    [
        UniformCrossover(),
        BLXAlphaCrossover(),
        SPXCrossover(),
        SBXCrossover(),
        VSBXCrossover(),
        UNDXCrossover(),
    ],
)
def test_crossover_numerical_distribution(crossover: BaseCrossover) -> None:
    study = optuna.study.create_study()
    rng = np.random.RandomState()
    search_space = {"x": FloatDistribution(1, 10), "y": IntDistribution(1, 10)}
    numerical_transform = _SearchSpaceTransform(search_space)
    parent_params = np.array([[1.0, 2], [3.0, 4]])

    if crossover.n_parents == 3:
        parent_params = np.append(parent_params, [[5.0, 6]], axis=0)

    child_params = crossover.crossover(parent_params, rng, study, numerical_transform.bounds)
    assert child_params.ndim == 1
    assert len(child_params) == len(search_space)
    assert not any(np.isnan(child_params))
    assert not any(np.isinf(child_params))


def test_crossover_inlined_categorical_distribution() -> None:
    search_space: Dict[str, BaseDistribution] = {
        "x": CategoricalDistribution(choices=["a", "c"]),
        "y": CategoricalDistribution(choices=["b", "d"]),
    }
    parent_params = np.array([["a", "b"], ["c", "d"]])
    rng = np.random.RandomState()
    child_params = _inlined_categorical_uniform_crossover(parent_params, rng, 0.5, search_space)

    assert child_params.ndim == 1
    assert len(child_params) == len(search_space)
    assert all([isinstance(param, str) for param in child_params])

    # Is child param from correct distribution?
    search_space["x"].to_internal_repr(child_params[0])
    search_space["y"].to_internal_repr(child_params[1])


@pytest.mark.parametrize(
    "crossover",
    [
        UniformCrossover(),
        BLXAlphaCrossover(),
        SPXCrossover(),
        SBXCrossover(),
        VSBXCrossover(),
        UNDXCrossover(),
    ],
)
def test_crossover_duplicated_param_values(crossover: BaseCrossover) -> None:
    param_values = [1.0, 2.0]

    study = optuna.study.create_study()
    rng = np.random.RandomState()
    search_space = {"x": FloatDistribution(1, 10), "y": IntDistribution(1, 10)}
    numerical_transform = _SearchSpaceTransform(search_space)
    parent_params = np.array([param_values, param_values])

    if crossover.n_parents == 3:
        parent_params = np.append(parent_params, [param_values], axis=0)

    child_params = crossover.crossover(parent_params, rng, study, numerical_transform.bounds)
    assert child_params.ndim == 1
    np.testing.assert_almost_equal(child_params, param_values)


@pytest.mark.parametrize(
    "crossover,rand_value,expected_params",
    [
        (UniformCrossover(), 0.0, np.array([1.0, 2.0])),  # p1.
        (UniformCrossover(), 0.5, np.array([3.0, 4.0])),  # p2.
        (UniformCrossover(), 1.0, np.array([3.0, 4.0])),  # p2.
        (BLXAlphaCrossover(), 0.0, np.array([0.0, 1.0])),  # p1 - [1, 1].
        (BLXAlphaCrossover(), 0.5, np.array([2.0, 3.0])),  # (p1 + p2) / 2.
        (BLXAlphaCrossover(), 1.0, np.array([4.0, 5.0])),  # p2 + [1, 1].
        # G = [3, 4], xks=[[-1, 0], [3, 4]. [7, 8]].
        (SPXCrossover(), 0.0, np.array([7, 8])),  # rs = [0, 0], xks[-1].
        (SPXCrossover(), 0.5, np.array([2.75735931, 3.75735931])),  # rs = [0.5, 0.25].
        (SPXCrossover(), 1.0, np.array([-1.0, 0.0])),  # rs = [1, 1], xks[0].
        (SBXCrossover(), 0.0, np.array([2.0, 3.0])),  # c1 = (p1 + p2) / 2.
        (SBXCrossover(), 0.5, np.array([3.0, 4.0])),  # p2.
        (SBXCrossover(), 1.0, np.array([3.0, 4.0])),  # p2.
        (VSBXCrossover(), 0.0, np.array([2.0, 3.0])),  # c1 = (p1 + p2) / 2.
        (VSBXCrossover(), 0.5, np.array([3.0, 4.0])),  # p2.
        (VSBXCrossover(), 1.0, np.array([3.0, 4.0])),  # p2.
        # p1, p2 and p3 are on x + 1, and distance from child to PSL is 0.
        (UNDXCrossover(), -0.5, np.array([3.0, 4.0])),  # [2, 3] + [-1, -1] + [0, 0].
        (UNDXCrossover(), 0.0, np.array([2.0, 3.0])),  # [2, 3] + [0, 0] + [0, 0].
        (UNDXCrossover(), 0.5, np.array([1.0, 2.0])),  # [2, 3] + [-1, -1] + [0, 0].
    ],
)
def test_crossover_deterministic(
    crossover: BaseCrossover, rand_value: float, expected_params: np.ndarray
) -> None:
    study = optuna.study.create_study()
    search_space: Dict[str, BaseDistribution] = {
        "x": FloatDistribution(1, 10),
        "y": FloatDistribution(1, 10),
    }
    numerical_transform = _SearchSpaceTransform(search_space)
    parent_params = np.array([[1.0, 2.0], [3.0, 4.0]])

    if crossover.n_parents == 3:
        parent_params = np.append(parent_params, [[5.0, 6.0]], axis=0)

    def _rand(*args: Any, **kwargs: Any) -> Any:
        if len(args) == 0:
            return rand_value
        return np.full(args[0], rand_value)

    def _normal(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("size") is None:
            return rand_value
        return np.full(kwargs.get("size"), rand_value)  # type: ignore

    rng = Mock()
    rng.rand = Mock(side_effect=_rand)
    rng.normal = Mock(side_effect=_normal)
    child_params = crossover.crossover(parent_params, rng, study, numerical_transform.bounds)
    np.testing.assert_almost_equal(child_params, expected_params)


@pytest.mark.parametrize(
    "dims_and_dividing_parameter",
    [
        (2, 2, [[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]]),
        (2, 3, [[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]]),
        (
            3,
            2,
            [
                [0.0, 0.0, 2.0],
                [0.0, 1.0, 1.0],
                [0.0, 2.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
        ),
    ],
)
def test_reference_point(
    dims_and_dividing_parameter: Tuple[int, int, Sequence[Sequence[int]]]
) -> None:
    n_dims, dividing_parameter, expected_reference_points = dims_and_dividing_parameter
    actual_reference_points = sorted(
        generate_default_reference_point(n_dims, dividing_parameter).tolist()
    )
    assert actual_reference_points == expected_reference_points


def test_associate() -> None:
    population = np.array(
        [
            [1.0, 2.0, 3.0],
            [3.0, 1.0, 2.0],
            [2.0, 3.0, 1.0],
            [2.0, 2.0, 2.0],
            [4.0, 5.0, 6.0],
            [6.0, 4.0, 5.0],
            [5.0, 6.0, 4.0],
            [4.0, 4.0, 4.0],
        ]
    )
    n_dims = 3
    dividing_parameter = 2
    reference_points = generate_default_reference_point(n_dims, dividing_parameter)
    elite_population_num = 4
    reference_points_per_count, ref2pops = _associate(
        population, reference_points, elite_population_num
    )
    actual_reference_points_per_count = dict(
        zip(reference_points_per_count, map(lambda x: set(x), reference_points_per_count.values()))
    )
    expected_reference_points_per_count = {1: {2, 4}, 2: {1}}
    assert actual_reference_points_per_count == expected_reference_points_per_count

    actual_ref2pops = dict(zip(ref2pops, map(lambda x: set(x), ref2pops.values())))
    expected_ref2pops = {
        1: {(4.0, 3), (4.06201920231798, 2)},
        2: {(4.06201920231798, 1)},
        4: {(4.06201920231798, 0)},
    }
    assert actual_ref2pops == expected_ref2pops


def test_niching() -> None:
    sampler = NSGAIIISampler(np.array([1.0]))
    target_population_size = 2
    population = [
        create_trial(values=[4.0, 5.0, 6.0]),
        create_trial(values=[6.0, 4.0, 5.0]),
        create_trial(values=[5.0, 6.0, 4.0]),
        create_trial(values=[4.0, 4.0, 4.0]),
    ]
    reference_points_per_count = defaultdict(list, {0: [1], 1: [2, 4]})
    ref2pops = defaultdict(
        list,
        {
            1: [(4.0, 3), (4.06201920231798, 2)],
            2: [(4.06201920231798, 1)],
            4: [(4.06201920231798, 0)],
        },
    )
    actual_additional_elite_population = _niching(
        target_population_size, population, reference_points_per_count, ref2pops, sampler._rng
    )
    expected_additional_elite_population = [population[3], population[1]]
    assert actual_additional_elite_population == expected_additional_elite_population
