from collections import Counter
from collections import defaultdict
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from unittest.mock import patch
import warnings

import numpy as np
import pytest

import optuna
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
from optuna.samplers.nsgaiii import _associate_individuals_with_reference_points
from optuna.samplers.nsgaiii import _POPULATION_CACHE_KEY_PREFIX
from optuna.samplers.nsgaiii import _preserve_niche_individuals
from optuna.samplers.nsgaiii import generate_default_reference_point
from optuna.trial import create_trial
from optuna.trial import FrozenTrial


def _nan_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True

    return a == b


def test_population_size() -> None:
    # Set `population_size` to 10.
    sampler = NSGAIIISampler(n_objectives=1, population_size=10)

    study = optuna.create_study(directions=["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float("x", 0, 9)], n_trials=40)

    generations = Counter(
        [t.system_attrs[optuna.samplers.nsgaiii._GENERATION_KEY] for t in study.trials]
    )
    assert generations == {0: 10, 1: 10, 2: 10, 3: 10}

    # Set `population_size` to 2.
    sampler = NSGAIIISampler(n_objectives=1, population_size=2)

    study = optuna.create_study(directions=["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float("x", 0, 9)], n_trials=40)

    generations = Counter(
        [t.system_attrs[optuna.samplers.nsgaiii._GENERATION_KEY] for t in study.trials]
    )
    assert generations == {i: 2 for i in range(20)}

    # Invalid population size.
    with pytest.raises(ValueError):
        # Less than 2.
        NSGAIIISampler(n_objectives=1, population_size=1)

    with pytest.raises(TypeError):
        # Not an integer.
        NSGAIIISampler(n_objectives=1, population_size=2.5)  # type: ignore[arg-type]


def test_mutation_prob() -> None:
    NSGAIIISampler(n_objectives=1, mutation_prob=None)
    NSGAIIISampler(n_objectives=1, mutation_prob=0.0)
    NSGAIIISampler(n_objectives=1, mutation_prob=0.5)
    NSGAIIISampler(n_objectives=1, mutation_prob=1.0)

    with pytest.raises(ValueError):
        NSGAIIISampler(n_objectives=1, mutation_prob=-0.5)

    with pytest.raises(ValueError):
        NSGAIIISampler(n_objectives=1, mutation_prob=1.1)


def test_crossover_prob() -> None:
    NSGAIIISampler(n_objectives=1, crossover_prob=0.0)
    NSGAIIISampler(n_objectives=1, crossover_prob=0.5)
    NSGAIIISampler(n_objectives=1, crossover_prob=1.0)

    with pytest.raises(ValueError):
        NSGAIIISampler(n_objectives=1, crossover_prob=-0.5)

    with pytest.raises(ValueError):
        NSGAIIISampler(n_objectives=1, crossover_prob=1.1)


def test_swapping_prob() -> None:
    NSGAIIISampler(n_objectives=1, swapping_prob=0.0)
    NSGAIIISampler(n_objectives=1, swapping_prob=0.5)
    NSGAIIISampler(n_objectives=1, swapping_prob=1.0)

    with pytest.raises(ValueError):
        NSGAIIISampler(n_objectives=1, swapping_prob=-0.5)

    with pytest.raises(ValueError):
        NSGAIIISampler(n_objectives=1, swapping_prob=1.1)


def test_constraints_func_none() -> None:
    n_trials = 4
    n_objectives = 2

    sampler = NSGAIIISampler(n_objectives=n_objectives, population_size=2)

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
        sampler = NSGAIIISampler(
            n_objectives=n_objectives, population_size=2, constraints_func=constraints_func
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
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = NSGAIIISampler(
            n_objectives=n_objectives, population_size=2, constraints_func=constraints_func
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


def test_study_system_attr_for_population_cache() -> None:
    sampler = NSGAIIISampler(n_objectives=1, population_size=10)
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
        NSGAIIISampler(n_objectives=1, constraints_func=lambda _: [0])


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
    sampler = NSGAIIISampler(n_objectives=1)
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
    sampler = NSGAIIISampler(n_objectives=n_objectives, population_size=population_size)

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
    sampler = NSGAIIISampler(n_objectives=n_objectives, population_size=population_size)

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
        NSGAIIISampler(n_objectives=1, population_size=population_size, crossover=crossover)


@pytest.mark.parametrize(
    "n_objectives,dividing_parameter,expected_reference_points",
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
    n_objectives: int, dividing_parameter: int, expected_reference_points: Sequence[Sequence[int]]
) -> None:
    actual_reference_points = sorted(
        generate_default_reference_point(n_objectives, dividing_parameter).tolist()
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
    reference_points = generate_default_reference_point(n_objectives=3, dividing_parameter=2)
    elite_population_num = 4
    (
        nearest_points_count_to_reference_point,
        reference_point_to_population,
    ) = _associate_individuals_with_reference_points(
        population, reference_points, elite_population_num
    )
    actual_reference_points_per_count = dict(
        zip(
            nearest_points_count_to_reference_point,
            map(lambda x: set(x), nearest_points_count_to_reference_point.values()),
        )
    )
    expected_reference_points_per_count = {1: {2, 4}, 2: {1}}
    assert actual_reference_points_per_count == expected_reference_points_per_count

    actual_reference_point_to_population = dict(
        zip(
            reference_point_to_population,
            map(lambda x: set(x), reference_point_to_population.values()),
        )
    )
    expected_reference_point_to_population = {
        1: {(4.0, 3), (4.06201920231798, 2)},
        2: {(4.06201920231798, 1)},
        4: {(4.06201920231798, 0)},
    }
    assert actual_reference_point_to_population == expected_reference_point_to_population


def test_niching() -> None:
    sampler = NSGAIIISampler(n_objectives=3, seed=42)
    target_population_size = 2
    population = [
        create_trial(values=[4.0, 5.0, 6.0]),
        create_trial(values=[6.0, 4.0, 5.0]),
        create_trial(values=[5.0, 6.0, 4.0]),
        create_trial(values=[4.0, 4.0, 4.0]),
    ]
    # each reference point 2 and 4 have an elite individual.
    nearest_points_count_to_reference_point = defaultdict(list, {0: [1], 1: [2, 4]})
    reference_point_to_population = defaultdict(
        list,
        {
            1: [(4.0, 3), (4.06201920231798, 2)],
            2: [(4.06201920231798, 1)],
            4: [(4.06201920231798, 0)],
        },
    )
    actual_additional_elite_population = [
        trial.values
        for trial in _preserve_niche_individuals(
            target_population_size,
            population,
            nearest_points_count_to_reference_point,
            reference_point_to_population,
            sampler._rng,
        )
    ]

    expected_additional_elite_population = [population[3].values, population[2].values]
    assert actual_additional_elite_population == expected_additional_elite_population
