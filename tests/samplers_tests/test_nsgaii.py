from collections import Counter
from typing import Any
from typing import Callable
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
from optuna.samplers import NSGAIISampler
from optuna.samplers.nsgaii import BaseCrossover
from optuna.samplers.nsgaii import BLXAlphaCrossover
from optuna.samplers.nsgaii import SBXCrossover
from optuna.samplers.nsgaii import SPXCrossover
from optuna.samplers.nsgaii import UNDXCrossover
from optuna.samplers.nsgaii import UniformCrossover
from optuna.samplers.nsgaii import VSBXCrossover
from optuna.samplers.nsgaii._crossover import _inlined_categorical_uniform_crossover
from optuna.samplers.nsgaii._sampler import _CONSTRAINTS_KEY
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial


def _nan_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True

    return a == b


def test_population_size() -> None:
    # Set `population_size` to 10.
    sampler = NSGAIISampler(population_size=10)

    study = optuna.create_study(directions=["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float("x", 0, 9)], n_trials=40)

    generations = Counter(
        [t.system_attrs[optuna.samplers.nsgaii._sampler._GENERATION_KEY] for t in study.trials]
    )
    assert generations == {0: 10, 1: 10, 2: 10, 3: 10}

    # Set `population_size` to 2.
    sampler = NSGAIISampler(population_size=2)

    study = optuna.create_study(directions=["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float("x", 0, 9)], n_trials=40)

    generations = Counter(
        [t.system_attrs[optuna.samplers.nsgaii._sampler._GENERATION_KEY] for t in study.trials]
    )
    assert generations == {i: 2 for i in range(20)}

    # Invalid population size.
    with pytest.raises(ValueError):
        # Less than 2.
        NSGAIISampler(population_size=1)

    with pytest.raises(TypeError):
        # Not an integer.
        NSGAIISampler(population_size=2.5)  # type: ignore


def test_mutation_prob() -> None:
    NSGAIISampler(mutation_prob=None)
    NSGAIISampler(mutation_prob=0.0)
    NSGAIISampler(mutation_prob=0.5)
    NSGAIISampler(mutation_prob=1.0)

    with pytest.raises(ValueError):
        NSGAIISampler(mutation_prob=-0.5)

    with pytest.raises(ValueError):
        NSGAIISampler(mutation_prob=1.1)


def test_crossover_prob() -> None:
    NSGAIISampler(crossover_prob=0.0)
    NSGAIISampler(crossover_prob=0.5)
    NSGAIISampler(crossover_prob=1.0)

    with pytest.raises(ValueError):
        NSGAIISampler(crossover_prob=-0.5)

    with pytest.raises(ValueError):
        NSGAIISampler(crossover_prob=1.1)


def test_swapping_prob() -> None:
    NSGAIISampler(swapping_prob=0.0)
    NSGAIISampler(swapping_prob=0.5)
    NSGAIISampler(swapping_prob=1.0)

    with pytest.raises(ValueError):
        NSGAIISampler(swapping_prob=-0.5)

    with pytest.raises(ValueError):
        NSGAIISampler(swapping_prob=1.1)


def test_constraints_func_none() -> None:
    n_trials = 4
    n_objectives = 2

    sampler = NSGAIISampler(population_size=2)

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler)
    study.optimize(
        lambda t: [t.suggest_float(f"x{i}", 0, 1) for i in range(n_objectives)], n_trials=n_trials
    )

    assert len(study.trials) == n_trials
    for trial in study.trials:
        assert _CONSTRAINTS_KEY not in trial.system_attrs


@pytest.mark.parametrize(
    "constraint_value", [-1.0, 0.0, 1.0, -float("inf"), float("inf"), float("nan")]
)
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
        sampler = NSGAIISampler(population_size=2, constraints_func=constraints_func)

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler)
    study.optimize(
        lambda t: [t.suggest_float(f"x{i}", 0, 1) for i in range(n_objectives)], n_trials=n_trials
    )

    assert len(study.trials) == n_trials
    assert constraints_func_call_count == n_trials
    for trial in study.trials:
        for x, y in zip(trial.system_attrs[_CONSTRAINTS_KEY], (constraint_value + trial.number,)):
            assert _nan_equal(x, y)


def test_fast_non_dominated_sort() -> None:
    sampler = NSGAIISampler()

    # Single objective.
    directions = [StudyDirection.MINIMIZE]
    trials = [
        _create_frozen_trial(0, [10]),
        _create_frozen_trial(1, [20]),
        _create_frozen_trial(2, [20]),
        _create_frozen_trial(3, [30]),
    ]
    population_per_rank = sampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {0},
        {1, 2},
        {3},
    ]

    # Two objective.
    directions = [StudyDirection.MAXIMIZE, StudyDirection.MAXIMIZE]
    trials = [
        _create_frozen_trial(0, [10, 30]),
        _create_frozen_trial(1, [10, 10]),
        _create_frozen_trial(2, [20, 20]),
        _create_frozen_trial(3, [30, 10]),
        _create_frozen_trial(4, [15, 15]),
    ]
    population_per_rank = sampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {0, 2, 3},
        {4},
        {1},
    ]

    # Three objective.
    directions = [StudyDirection.MAXIMIZE, StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]
    trials = [
        _create_frozen_trial(0, [5, 5, 4]),
        _create_frozen_trial(1, [5, 5, 5]),
        _create_frozen_trial(2, [9, 9, 0]),
        _create_frozen_trial(3, [5, 7, 5]),
        _create_frozen_trial(4, [0, 0, 9]),
        _create_frozen_trial(5, [0, 9, 9]),
    ]
    population_per_rank = sampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {2},
        {0, 3, 5},
        {1},
        {4},
    ]


def test_fast_non_dominated_sort_constrained_feasible() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = NSGAIISampler(constraints_func=lambda _: [0])

    # Single objective.
    directions = [StudyDirection.MINIMIZE]
    trials = [
        _create_frozen_trial(0, [10], [0]),
        _create_frozen_trial(1, [20], [0]),
        _create_frozen_trial(2, [20], [0]),
        _create_frozen_trial(3, [30], [0]),
    ]
    population_per_rank = sampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {0},
        {1, 2},
        {3},
    ]

    # Two objective.
    directions = [StudyDirection.MAXIMIZE, StudyDirection.MAXIMIZE]
    trials = [
        _create_frozen_trial(0, [10, 30], [0]),
        _create_frozen_trial(1, [10, 10], [0]),
        _create_frozen_trial(2, [20, 20], [0]),
        _create_frozen_trial(3, [30, 10], [0]),
        _create_frozen_trial(4, [15, 15], [0]),
    ]
    population_per_rank = sampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {0, 2, 3},
        {4},
        {1},
    ]

    # Three objective.
    directions = [StudyDirection.MAXIMIZE, StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]
    trials = [
        _create_frozen_trial(0, [5, 5, 4], [0]),
        _create_frozen_trial(1, [5, 5, 5], [0]),
        _create_frozen_trial(2, [9, 9, 0], [0]),
        _create_frozen_trial(3, [5, 7, 5], [0]),
        _create_frozen_trial(4, [0, 0, 9], [0]),
        _create_frozen_trial(5, [0, 9, 9], [0]),
    ]
    population_per_rank = sampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {2},
        {0, 3, 5},
        {1},
        {4},
    ]


def test_fast_non_dominated_sort_constrained_infeasible() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = NSGAIISampler(constraints_func=lambda _: [0])

    # Single objective.
    directions = [StudyDirection.MINIMIZE]
    trials = [
        _create_frozen_trial(0, [10], [1]),
        _create_frozen_trial(1, [20], [3]),
        _create_frozen_trial(2, [20], [2]),
        _create_frozen_trial(3, [30], [1]),
    ]
    population_per_rank = sampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {0, 3},
        {2},
        {1},
    ]

    # Two objective.
    directions = [StudyDirection.MAXIMIZE, StudyDirection.MAXIMIZE]
    trials = [
        _create_frozen_trial(0, [10, 30], [1, 1]),
        _create_frozen_trial(1, [10, 10], [2, 2]),
        _create_frozen_trial(2, [20, 20], [3, 3]),
        _create_frozen_trial(3, [30, 10], [2, 4]),
        _create_frozen_trial(4, [15, 15], [4, 4]),
    ]
    population_per_rank = sampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {0},
        {1},
        {2, 3},
        {4},
    ]

    # Three objective.
    directions = [StudyDirection.MAXIMIZE, StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]
    trials = [
        _create_frozen_trial(0, [5, 5, 4], [1, 1, 1]),
        _create_frozen_trial(1, [5, 5, 5], [1, 1, 1]),
        _create_frozen_trial(2, [9, 9, 0], [3, 3, 3]),
        _create_frozen_trial(3, [5, 7, 5], [2, 2, 2]),
        _create_frozen_trial(4, [0, 0, 9], [1, 1, 4]),
        _create_frozen_trial(5, [0, 9, 9], [2, 1, 3]),
    ]
    population_per_rank = sampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {0, 1},
        {3, 4, 5},
        {2},
    ]


def test_fast_non_dominated_sort_constrained_feasible_infeasible() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = NSGAIISampler(constraints_func=lambda _: [0])

    # Single objective.
    directions = [StudyDirection.MINIMIZE]
    trials = [
        _create_frozen_trial(0, [10], [0]),
        _create_frozen_trial(1, [20], [-1]),
        _create_frozen_trial(2, [20], [-2]),
        _create_frozen_trial(3, [30], [1]),
    ]
    population_per_rank = sampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {0},
        {1, 2},
        {3},
    ]

    # Two objective.
    directions = [StudyDirection.MAXIMIZE, StudyDirection.MAXIMIZE]
    trials = [
        _create_frozen_trial(0, [10, 30], [-1, -1]),
        _create_frozen_trial(1, [10, 10], [-2, -2]),
        _create_frozen_trial(2, [20, 20], [3, 3]),
        _create_frozen_trial(3, [30, 10], [6, -1]),
        _create_frozen_trial(4, [15, 15], [4, 4]),
    ]
    population_per_rank = sampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {0},
        {1},
        {2, 3},
        {4},
    ]

    # Three objective.
    directions = [StudyDirection.MAXIMIZE, StudyDirection.MAXIMIZE, StudyDirection.MINIMIZE]
    trials = [
        _create_frozen_trial(0, [5, 5, 4], [-1, -1, -1]),
        _create_frozen_trial(1, [5, 5, 5], [1, 1, -1]),
        _create_frozen_trial(2, [9, 9, 0], [1, -1, -1]),
        _create_frozen_trial(3, [5, 7, 5], [-1, -1, -1]),
        _create_frozen_trial(4, [0, 0, 9], [-1, -1, -1]),
        _create_frozen_trial(5, [0, 9, 9], [-1, -1, -1]),
    ]
    population_per_rank = sampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {0, 3, 5},
        {4},
        {2},
        {1},
    ]


def test_fast_non_dominated_sort_missing_constraint_values() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = NSGAIISampler(constraints_func=lambda _: [0])

    # Single objective.
    directions = [StudyDirection.MINIMIZE]
    trials = [
        _create_frozen_trial(0, [10]),
        _create_frozen_trial(1, [20]),
        _create_frozen_trial(2, [20], [0]),
        _create_frozen_trial(3, [20], [1]),
        _create_frozen_trial(4, [30], [-1]),
    ]
    with pytest.warns(UserWarning):
        population_per_rank = sampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {2},
        {4},
        {3},
        {0},
        {1},
    ]

    # Two objectives.
    directions = [StudyDirection.MAXIMIZE, StudyDirection.MAXIMIZE]
    trials = [
        _create_frozen_trial(0, [50, 30]),
        _create_frozen_trial(1, [30, 50]),
        _create_frozen_trial(2, [20, 20], [3, 3]),
        _create_frozen_trial(3, [30, 10], [0, -1]),
        _create_frozen_trial(4, [15, 15], [4, 4]),
    ]
    with pytest.warns(UserWarning):
        population_per_rank = sampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {3},
        {2},
        {4},
        {0, 1},
    ]


@pytest.mark.parametrize(
    "values, expected_dist",
    [
        ([[5], [6], [9], [0]], [6 / 9, 4 / 9, float("inf"), float("inf")]),
        ([[5, 0], [6, 0], [9, 0], [0, 0]], [6 / 9, 4 / 9, float("inf"), float("inf")]),
        (
            [[5, -1], [6, 0], [9, 1], [0, 2]],
            [float("inf"), 4 / 9 + 2 / 3, float("inf"), float("inf")],
        ),
        ([[5]], [0]),
        ([[5], [5]], [0, 0]),
        (
            [[1], [2], [float("inf")]],
            [float("inf"), float("nan"), float("inf")],
        ),  # TODO(knshnb): Decide expected behavior for this case.
        (
            [[float("-inf")], [1], [2]],
            [float("inf"), float("nan"), float("inf")],
        ),  # TODO(knshnb): Decide expected behavior for this case.
    ],
)
def test_calc_crowding_distance(values: List[List[float]], expected_dist: List[float]) -> None:
    trials = [_create_frozen_trial(i, value) for i, value in enumerate(values)]
    crowding_dist = optuna.samplers.nsgaii._sampler._calc_crowding_distance(trials)
    for i in range(len(trials)):
        assert _nan_equal(crowding_dist[i], expected_dist[i]), i


@pytest.mark.parametrize(
    "values",
    [
        [[5], [6], [9], [0]],
        [[5, 0], [6, 0], [9, 0], [0, 0]],
        [[5, -1], [6, 0], [9, 1], [0, 2]],
        [[1], [2], [float("inf")]],
        [[float("-inf")], [1], [2]],
    ],
)
def test_crowding_distance_sort(values: List[List[float]]) -> None:
    """Checks that trials are sorted by the values of `_calc_crowding_distance`."""
    trials = [_create_frozen_trial(i, value) for i, value in enumerate(values)]
    crowding_dist = optuna.samplers.nsgaii._sampler._calc_crowding_distance(trials)
    optuna.samplers.nsgaii._sampler._crowding_distance_sort(trials)
    sorted_dist = [crowding_dist[t.number] for t in trials]
    assert sorted_dist == sorted(sorted_dist, reverse=True)


def test_study_system_attr_for_population_cache() -> None:
    sampler = NSGAIISampler(population_size=10)
    study = optuna.create_study(directions=["minimize"], sampler=sampler)

    def get_cached_entries(
        study: optuna.study.Study,
    ) -> List[Tuple[int, List[int]]]:
        return [
            v
            for k, v in study.system_attrs.items()
            if k.startswith(optuna.samplers.nsgaii._sampler._POPULATION_CACHE_KEY_PREFIX)
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
        NSGAIISampler(constraints_func=lambda _: [0])


# TODO(ohta): Consider to move this utility function to `optuna.testing` module.
def _create_frozen_trial(
    number: int, values: List[float], constraints: Optional[List[float]] = None
) -> optuna.trial.FrozenTrial:
    trial = optuna.trial.create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        values=values,
        system_attrs={} if constraints is None else {_CONSTRAINTS_KEY: constraints},
    )
    trial.number = number
    trial._trial_id = number
    return trial


def test_call_after_trial_of_random_sampler() -> None:
    sampler = NSGAIISampler()
    study = optuna.create_study(sampler=sampler)
    with patch.object(
        sampler._random_sampler, "after_trial", wraps=sampler._random_sampler.after_trial
    ) as mock_object:
        study.optimize(lambda _: 1.0, n_trials=1)
        assert mock_object.call_count == 1


parametrize_nsga2_sampler = pytest.mark.parametrize(
    "sampler_class",
    [
        lambda: NSGAIISampler(population_size=2, crossover=UniformCrossover()),
        lambda: NSGAIISampler(population_size=2, crossover=BLXAlphaCrossover()),
        lambda: NSGAIISampler(population_size=2, crossover=SBXCrossover()),
        lambda: NSGAIISampler(population_size=2, crossover=VSBXCrossover()),
        lambda: NSGAIISampler(population_size=3, crossover=UNDXCrossover()),
        lambda: NSGAIISampler(population_size=3, crossover=UNDXCrossover()),
    ],
)


@parametrize_nsga2_sampler
@pytest.mark.parametrize("n_objectives", [1, 2, 3])
def test_crossover_objectives(n_objectives: int, sampler_class: Callable[[], BaseSampler]) -> None:
    n_trials = 8

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler_class())
    study.optimize(
        lambda t: [t.suggest_float(f"x{i}", 0, 1) for i in range(n_objectives)], n_trials=n_trials
    )

    assert len(study.trials) == n_trials


@parametrize_nsga2_sampler
@pytest.mark.parametrize("n_params", [1, 2, 3])
def test_crossover_dims(n_params: int, sampler_class: Callable[[], BaseSampler]) -> None:
    def objective(trial: optuna.Trial) -> float:
        xs = [trial.suggest_float(f"x{dim}", -10, 10) for dim in range(n_params)]
        return sum(xs)

    n_objectives = 1
    n_trials = 8

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler_class())
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
        NSGAIISampler(population_size=population_size, crossover=crossover)


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
