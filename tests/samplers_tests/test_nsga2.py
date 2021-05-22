from collections import Counter
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from unittest.mock import patch
import warnings

import pytest

import optuna
from optuna._study_direction import StudyDirection
from optuna.samplers import NSGAIISampler
from optuna.samplers._nsga2 import _CONSTRAINTS_KEY
from optuna.trial import FrozenTrial


def test_population_size() -> None:
    # Set `population_size` to 10.
    sampler = NSGAIISampler(population_size=10)

    study = optuna.create_study(directions=["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_uniform("x", 0, 9)], n_trials=40)

    generations = Counter(
        [t.system_attrs[optuna.samplers._nsga2._GENERATION_KEY] for t in study.trials]
    )
    assert generations == {0: 10, 1: 10, 2: 10, 3: 10}

    # Set `population_size` to 2.
    sampler = NSGAIISampler(population_size=2)

    study = optuna.create_study(directions=["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_uniform("x", 0, 9)], n_trials=40)

    generations = Counter(
        [t.system_attrs[optuna.samplers._nsga2._GENERATION_KEY] for t in study.trials]
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


def test_constraints_func() -> None:
    n_trials = 4
    n_objectives = 2
    constraints_func_call_count = 0

    def constraints_func(trial: FrozenTrial) -> Sequence[float]:
        nonlocal constraints_func_call_count
        constraints_func_call_count += 1

        return (trial.number,)

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
        assert trial.system_attrs[_CONSTRAINTS_KEY] == (trial.number,)


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


def test_crowding_distance_sort() -> None:
    trials = [
        _create_frozen_trial(0, [5]),
        _create_frozen_trial(1, [6]),
        _create_frozen_trial(2, [9]),
        _create_frozen_trial(3, [0]),
    ]
    optuna.samplers._nsga2._crowding_distance_sort(trials)
    assert [t.number for t in trials] == [2, 3, 0, 1]

    trials = [
        _create_frozen_trial(0, [5, 0]),
        _create_frozen_trial(1, [6, 0]),
        _create_frozen_trial(2, [9, 0]),
        _create_frozen_trial(3, [0, 0]),
    ]
    optuna.samplers._nsga2._crowding_distance_sort(trials)
    assert [t.number for t in trials] == [2, 3, 0, 1]


def test_study_system_attr_for_population_cache() -> None:
    sampler = NSGAIISampler(population_size=10)
    study = optuna.create_study(directions=["minimize"], sampler=sampler)

    def get_cached_entries(
        study: optuna.study.Study,
    ) -> List[Tuple[int, List[int]]]:
        return [
            v
            for k, v in study.system_attrs.items()
            if k.startswith(optuna.samplers._nsga2._POPULATION_CACHE_KEY_PREFIX)
        ]

    study.optimize(lambda t: [t.suggest_uniform("x", 0, 9)], n_trials=10)
    cached_entries = get_cached_entries(study)
    assert len(cached_entries) == 0

    study.optimize(lambda t: [t.suggest_uniform("x", 0, 9)], n_trials=1)
    cached_entries = get_cached_entries(study)
    assert len(cached_entries) == 1
    assert cached_entries[0][0] == 0  # Cached generation.
    assert len(cached_entries[0][1]) == 10  # Population size.

    study.optimize(lambda t: [t.suggest_uniform("x", 0, 9)], n_trials=10)
    cached_entries = get_cached_entries(study)
    assert len(cached_entries) == 1
    assert cached_entries[0][0] == 1  # Cached generation.
    assert len(cached_entries[0][1]) == 10  # Population size.


def test_reseed_rng() -> None:
    sampler = NSGAIISampler(population_size=10)
    original_seed = sampler._rng.seed
    original_random_sampler_seed = sampler._random_sampler._rng.seed

    sampler.reseed_rng()
    assert original_seed != sampler._rng.seed
    assert original_random_sampler_seed != sampler._random_sampler._rng.seed


def test_constraints_func_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        NSGAIISampler(constraints_func=lambda _: [0])


# TODO(ohta): Consider to move this utility function to `optuna.testing` module.
def _create_frozen_trial(
    number: int, values: List[float], constraints: Optional[List[float]] = None
) -> optuna.trial.FrozenTrial:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
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
