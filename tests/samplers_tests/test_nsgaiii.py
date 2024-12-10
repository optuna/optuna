from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from unittest.mock import MagicMock
from unittest.mock import patch
import warnings

import numpy as np
import pytest

import optuna
from optuna.samplers import BaseSampler
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._nsgaiii._elite_population_selection_strategy import (
    _associate_individuals_with_reference_points,
)
from optuna.samplers._nsgaiii._elite_population_selection_strategy import (
    _generate_default_reference_point,
)
from optuna.samplers._nsgaiii._elite_population_selection_strategy import (
    _normalize_objective_values,
)
from optuna.samplers._nsgaiii._elite_population_selection_strategy import (
    _preserve_niche_individuals,
)
from optuna.samplers._nsgaiii._elite_population_selection_strategy import _COEF
from optuna.samplers._nsgaiii._elite_population_selection_strategy import _filter_inf
from optuna.samplers._nsgaiii._sampler import _POPULATION_CACHE_KEY_PREFIX
from optuna.samplers._nsgaiii._sampler import NSGAIIISampler
from optuna.samplers.nsgaii import BaseCrossover
from optuna.samplers.nsgaii import BLXAlphaCrossover
from optuna.samplers.nsgaii import SBXCrossover
from optuna.samplers.nsgaii import SPXCrossover
from optuna.samplers.nsgaii import UNDXCrossover
from optuna.samplers.nsgaii import UniformCrossover
from optuna.samplers.nsgaii import VSBXCrossover
from optuna.samplers.nsgaii._after_trial_strategy import NSGAIIAfterTrialStrategy
from optuna.trial import create_trial
from optuna.trial import FrozenTrial


def test_population_size() -> None:
    # Set `population_size` to 10.
    sampler = NSGAIIISampler(population_size=10)

    study = optuna.create_study(directions=["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float("x", 0, 9)], n_trials=40)

    generations = Counter(
        [t.system_attrs[optuna.samplers._nsgaiii._sampler._GENERATION_KEY] for t in study.trials]
    )
    assert generations == {0: 10, 1: 10, 2: 10, 3: 10}

    # Set `population_size` to 2.
    sampler = NSGAIIISampler(population_size=2)

    study = optuna.create_study(directions=["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float("x", 0, 9)], n_trials=40)

    generations = Counter(
        [t.system_attrs[optuna.samplers._nsgaiii._sampler._GENERATION_KEY] for t in study.trials]
    )
    assert generations == {i: 2 for i in range(20)}

    # Invalid population size.
    with pytest.raises(ValueError):
        # Less than 2.
        NSGAIIISampler(population_size=1)

    with pytest.raises(ValueError):
        mock_crossover = MagicMock(spec=BaseCrossover)
        mock_crossover.configure_mock(n_parents=3)
        NSGAIIISampler(population_size=2, crossover=mock_crossover)


def test_mutation_prob() -> None:
    NSGAIIISampler(mutation_prob=None)
    NSGAIIISampler(mutation_prob=0.0)
    NSGAIIISampler(mutation_prob=0.5)
    NSGAIIISampler(mutation_prob=1.0)

    with pytest.raises(ValueError):
        NSGAIIISampler(mutation_prob=-0.5)

    with pytest.raises(ValueError):
        NSGAIIISampler(mutation_prob=1.1)


def test_crossover_prob() -> None:
    NSGAIIISampler(crossover_prob=0.0)
    NSGAIIISampler(crossover_prob=0.5)
    NSGAIIISampler(crossover_prob=1.0)

    with pytest.raises(ValueError):
        NSGAIIISampler(crossover_prob=-0.5)

    with pytest.raises(ValueError):
        NSGAIIISampler(crossover_prob=1.1)


def test_swapping_prob() -> None:
    NSGAIIISampler(swapping_prob=0.0)
    NSGAIIISampler(swapping_prob=0.5)
    NSGAIIISampler(swapping_prob=1.0)

    with pytest.raises(ValueError):
        NSGAIIISampler(swapping_prob=-0.5)

    with pytest.raises(ValueError):
        NSGAIIISampler(swapping_prob=1.1)


def test_constraints_func_none() -> None:
    n_trials = 4
    n_objectives = 2

    sampler = NSGAIIISampler(population_size=2)

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
        sampler = NSGAIIISampler(population_size=2, constraints_func=constraints_func)

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
        sampler = NSGAIIISampler(population_size=2, constraints_func=constraints_func)

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
    sampler = NSGAIIISampler(population_size=10)
    study = optuna.create_study(directions=["minimize"], sampler=sampler)

    def get_cached_entries(
        study: optuna.study.Study,
    ) -> list[tuple[int, list[int]]]:
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
        NSGAIIISampler(constraints_func=lambda _: [0])


def test_child_generation_strategy_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        NSGAIIISampler(child_generation_strategy=lambda study, search_space, parent_population: {})


def test_after_trial_strategy_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        NSGAIIISampler(after_trial_strategy=lambda study, trial, state, value: None)


def test_call_after_trial_of_random_sampler() -> None:
    sampler = NSGAIIISampler()
    study = optuna.create_study(sampler=sampler)
    with patch.object(
        sampler._random_sampler, "after_trial", wraps=sampler._random_sampler.after_trial
    ) as mock_object:
        study.optimize(lambda _: 1.0, n_trials=1)
        assert mock_object.call_count == 1


def test_call_after_trial_of_after_trial_strategy() -> None:
    sampler = NSGAIIISampler()
    study = optuna.create_study(sampler=sampler)
    with patch.object(sampler, "_after_trial_strategy") as mock_object:
        study.optimize(lambda _: 1.0, n_trials=1)
        assert mock_object.call_count == 1


@patch("optuna.samplers.nsgaii._after_trial_strategy._process_constraints_after_trial")
def test_nsgaii_after_trial_strategy(mock_func: MagicMock) -> None:
    def constraints_func(_: FrozenTrial) -> Sequence[float]:
        return (float("nan"),)

    state = optuna.trial.TrialState.FAIL
    study = optuna.create_study()
    trial = optuna.trial.create_trial(state=state)

    after_trial_strategy_without_constrains = NSGAIIAfterTrialStrategy()
    after_trial_strategy_without_constrains(study, trial, state)
    assert mock_func.call_count == 0

    after_trial_strategy_with_constrains = NSGAIIAfterTrialStrategy(
        constraints_func=constraints_func
    )
    after_trial_strategy_with_constrains(study, trial, state)
    assert mock_func.call_count == 1


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
    sampler = NSGAIIISampler(population_size=population_size)

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler)
    study.optimize(
        lambda t: [t.suggest_float(f"x{i}", 0, 1) for i in range(n_objectives)],
        n_trials=n_trials,
    )
    assert len(study.trials) == n_trials


@parametrize_crossover_population
@pytest.mark.parametrize("n_params", [1, 2, 3])
def test_crossover_dims(n_params: int, crossover: BaseSampler, population_size: int) -> None:
    def objective(trial: optuna.Trial) -> float:
        xs = [trial.suggest_float(f"x{dim}", -10, 10) for dim in range(n_params)]
        return sum(xs)

    n_trials = 8
    sampler = NSGAIIISampler(population_size=population_size)

    study = optuna.create_study(directions=["minimize"], sampler=sampler)
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
        NSGAIIISampler(population_size=population_size, crossover=crossover)


@pytest.mark.parametrize(
    "n_objectives,dividing_parameter,expected_reference_points",
    [
        (1, 3, [[3.0]]),
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
def test_generate_reference_point(
    n_objectives: int, dividing_parameter: int, expected_reference_points: Sequence[Sequence[int]]
) -> None:
    actual_reference_points = _generate_default_reference_point(n_objectives, dividing_parameter)
    order = np.lexsort([actual_reference_points[:, -i - 1] for i in range(n_objectives)])
    sorted_actual_reference_points = actual_reference_points[order]
    assert np.allclose(sorted_actual_reference_points, expected_reference_points)


@pytest.mark.parametrize(
    "objective_values, expected_normalized_value",
    [
        (
            [
                [1.0, 2.0],
                [float("inf"), 0.5],
            ],
            [
                [1.0, 2.0],
                [1.0, 0.5],
            ],
        ),
        (
            [
                [1.0, float("inf")],
                [float("inf"), 0.5],
            ],
            [
                [1.0, 0.5],
                [1.0, 0.5],
            ],
        ),
        (
            [
                [1.0, float("inf")],
                [3.0, 1.0],
                [2.0, 3.0],
                [float("inf"), 0.5],
            ],
            [
                [1.0, 3.0 + _COEF * 2.5],
                [3.0, 1.0],
                [2.0, 3.0],
                [3.0 + _COEF * 2.0, 0.5],
            ],
        ),
        (
            [
                [2.0, 3.0],
                [-float("inf"), 3.5],
            ],
            [
                [2.0, 3.0],
                [2.0, 3.5],
            ],
        ),
        (
            [
                [2.0, -float("inf")],
                [-float("inf"), 3.5],
            ],
            [
                [2.0, 3.5],
                [2.0, 3.5],
            ],
        ),
        (
            [
                [4.0, -float("inf")],
                [3.0, 1.0],
                [2.0, 3.0],
                [-float("inf"), 3.5],
            ],
            [
                [4.0, 1.0 - _COEF * 2.5],
                [3.0, 1.0],
                [2.0, 3.0],
                [2.0 - _COEF * 2.0, 3.5],
            ],
        ),
        (
            [
                [1.0, float("inf")],
                [3.0, -float("inf")],
                [float("inf"), 2.0],
                [-float("inf"), 3.5],
            ],
            [
                [1.0, 3.5 + _COEF * 1.5],
                [3.0, 2.0 - _COEF * 1.5],
                [3.0 + _COEF * 2.0, 2.0],
                [1.0 - _COEF * 2.0, 3.5],
            ],
        ),
    ],
)
def test_filter_inf(
    objective_values: Sequence[Sequence[int]], expected_normalized_value: Sequence[Sequence[int]]
) -> None:
    population = [create_trial(values=values) for values in objective_values]
    np.testing.assert_almost_equal(_filter_inf(population), np.array(expected_normalized_value))


@pytest.mark.parametrize(
    "objective_values, expected_normalized_value",
    [
        (
            [
                [2.71],
                [1.41],
                [3.14],
            ],
            [
                [(2.71 - 1.41) / (3.14 - 1.41)],
                [0],
                [1.0],
            ],
        ),
        (
            [
                [1.0, 2.0, 3.0],
                [3.0, 1.0, 2.0],
                [2.0, 3.0, 1.0],
                [2.0, 2.0, 2.0],
                [4.0, 5.0, 6.0],
                [6.0, 4.0, 5.0],
                [5.0, 6.0, 4.0],
                [4.0, 4.0, 4.0],
            ],
            [
                [0.0, 1.0 / 3.0, 2.0 / 3.0],
                [2.0 / 3.0, 0.0, 1.0 / 3.0],
                [1.0 / 3.0, 2.0 / 3.0, 0.0],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                [1.0, 4.0 / 3.0, 5.0 / 3.0],
                [5.0 / 3.0, 1.0, 4.0 / 3.0],
                [4.0 / 3.0, 5.0 / 3.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
        ),
        (
            [
                [1.0, 2.0, 3.0],
                [3.0, 1.0, 2.0],
            ],
            [
                [0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
        ),
    ],
)
def test_normalize(
    objective_values: Sequence[Sequence[int]], expected_normalized_value: Sequence[Sequence[int]]
) -> None:
    np.testing.assert_almost_equal(
        _normalize_objective_values(np.array(objective_values)),
        np.array(expected_normalized_value),
    )


@pytest.mark.parametrize(
    "objective_values, expected_indices, expected_distances",
    [
        ([[1.0], [2.0], [0.0], [3.0]], [0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0]),
        (
            [
                [1.0, 2.0, 3.0],
                [3.0, 1.0, 2.0],
                [2.0, 3.0, 1.0],
                [2.0, 2.0, 2.0],
                [4.0, 5.0, 6.0],
                [6.0, 4.0, 5.0],
                [5.0, 6.0, 4.0],
                [4.0, 4.0, 4.0],
                [0.0, 1.0, 10.0],
                [10.0, 0.0, 1.0],
                [1.0, 10.0, 0.0],
            ],
            [4, 2, 1, 1, 4, 2, 1, 1, 5, 0, 3],
            [
                1.22474487,
                1.22474487,
                1.22474487,
                2.0,
                4.0620192,
                4.0620192,
                4.0620192,
                4.0,
                1.0,
                1.0,
                1.0,
            ],
        ),
    ],
)
def test_associate(
    objective_values: Sequence[Sequence[float]],
    expected_indices: Sequence[int],
    expected_distances: Sequence[float],
) -> None:
    population = np.array(objective_values)
    n_objectives = population.shape[1]
    reference_points = _generate_default_reference_point(
        n_objectives=n_objectives, dividing_parameter=2
    )
    (
        closest_reference_points,
        distance_reference_points,
    ) = _associate_individuals_with_reference_points(population, reference_points)
    assert np.all(closest_reference_points == expected_indices)
    np.testing.assert_almost_equal(distance_reference_points, expected_distances)


@pytest.mark.parametrize(
    "population_value,closest_reference_points, distance_reference_points, "
    "expected_population_indices",
    [
        (
            [[1.0], [2.0], [0.0], [3.0], [3.5], [5.5], [1.2], [3.3], [4.8]],
            [0, 0, 0, 0, 0, 0, 0, 0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3, 0, 2, 4, 1],
        ),
        (
            [
                [4.0, 5.0, 6.0],
                [6.0, 4.0, 5.0],
                [5.0, 6.0, 4.0],
                [4.0, 4.0, 4.0],
                [0.0, 1.0, 10.0],
                [10.0, 0.0, 1.0],
                [1.0, 10.0, 0.0],
            ],
            [4, 2, 1, 1, 4, 2, 1, 1, 5, 0, 3],
            [
                1.22474487,
                1.22474487,
                1.22474487,
                2.0,
                4.0620192,
                4.0620192,
                4.0620192,
                4.0,
                1.0,
                1.0,
                1.0,
            ],
            [6, 5, 4, 0, 1],
        ),
    ],
)
def test_niching(
    population_value: Sequence[Sequence[float]],
    closest_reference_points: Sequence[int],
    distance_reference_points: Sequence[float],
    expected_population_indices: Sequence[int],
) -> None:
    sampler = NSGAIIISampler(seed=42)
    target_population_size = 5
    elite_population_num = 4
    population = [create_trial(values=value) for value in population_value]
    actual_additional_elite_population = [
        trial.values
        for trial in _preserve_niche_individuals(
            target_population_size,
            elite_population_num,
            population,
            np.array(closest_reference_points),
            np.array(distance_reference_points),
            sampler._rng.rng,
        )
    ]
    expected_additional_elite_population = [
        population[idx].values for idx in expected_population_indices
    ]
    assert np.all(actual_additional_elite_population == expected_additional_elite_population)


def test_niching_unexpected_target_population_size() -> None:
    sampler = NSGAIIISampler(seed=42)
    target_population_size = 2
    elite_population_num = 1
    population = [create_trial(values=[1.0])]
    with pytest.raises(ValueError):
        _preserve_niche_individuals(
            target_population_size,
            elite_population_num,
            population,
            np.array([0]),
            np.array([0.0]),
            sampler._rng.rng,
        )
