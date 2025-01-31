from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from collections.abc import Sequence
import itertools
from typing import Any
from unittest.mock import MagicMock
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
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers.nsgaii import BaseCrossover
from optuna.samplers.nsgaii import BLXAlphaCrossover
from optuna.samplers.nsgaii import SBXCrossover
from optuna.samplers.nsgaii import SPXCrossover
from optuna.samplers.nsgaii import UNDXCrossover
from optuna.samplers.nsgaii import UniformCrossover
from optuna.samplers.nsgaii import VSBXCrossover
from optuna.samplers.nsgaii._after_trial_strategy import NSGAIIAfterTrialStrategy
from optuna.samplers.nsgaii._child_generation_strategy import NSGAIIChildGenerationStrategy
from optuna.samplers.nsgaii._constraints_evaluation import _constrained_dominates
from optuna.samplers.nsgaii._constraints_evaluation import _validate_constraints
from optuna.samplers.nsgaii._crossover import _inlined_categorical_uniform_crossover
from optuna.samplers.nsgaii._elite_population_selection_strategy import (
    NSGAIIElitePopulationSelectionStrategy,
)
from optuna.samplers.nsgaii._elite_population_selection_strategy import _calc_crowding_distance
from optuna.samplers.nsgaii._elite_population_selection_strategy import _crowding_distance_sort
from optuna.samplers.nsgaii._elite_population_selection_strategy import _rank_population
from optuna.study._multi_objective import _dominates
from optuna.study._study_direction import StudyDirection
from optuna.testing.trials import _create_frozen_trial
from optuna.trial import FrozenTrial


def _nan_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True

    return a == b


def test_generation_key_name() -> None:
    assert NSGAIISampler._GENERATION_KEY == "NSGAIISampler:generation"


def test_population_size() -> None:
    # Set `population_size` to 10.
    sampler = NSGAIISampler(population_size=10)

    study = optuna.create_study(directions=["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float("x", 0, 9)], n_trials=40)

    generations = Counter([t.system_attrs[NSGAIISampler._GENERATION_KEY] for t in study.trials])
    assert generations == {0: 10, 1: 10, 2: 10, 3: 10}

    # Set `population_size` to 2.
    sampler = NSGAIISampler(population_size=2)

    study = optuna.create_study(directions=["minimize"], sampler=sampler)
    study.optimize(lambda t: [t.suggest_float("x", 0, 9)], n_trials=40)

    generations = Counter([t.system_attrs[NSGAIISampler._GENERATION_KEY] for t in study.trials])
    assert generations == {i: 2 for i in range(20)}

    # Invalid population size.
    with pytest.raises(ValueError):
        # Less than 2.
        NSGAIISampler(population_size=1)

    with pytest.raises(ValueError):
        mock_crossover = MagicMock(spec=BaseCrossover)
        mock_crossover.configure_mock(n_parents=3)
        NSGAIISampler(population_size=2, crossover=mock_crossover)


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

    with pytest.raises(ValueError):
        UniformCrossover(swapping_prob=-0.5)

    with pytest.raises(ValueError):
        UniformCrossover(swapping_prob=1.1)


@pytest.mark.parametrize("choices", [[-1, 0, 1], [True, False]])
def test_crossover_casting(choices: list[Any]) -> None:
    str_choices = list(map(str, choices))

    def objective(trial: optuna.Trial) -> Sequence[float]:
        cat_1 = trial.suggest_categorical("cat_1", choices)
        cat_2 = trial.suggest_categorical("cat_2", str_choices)
        assert isinstance(cat_1, type(choices[0]))
        assert isinstance(cat_2, type(str_choices[0]))
        return 1.0, 2.0

    population_size = 10
    sampler = NSGAIISampler(population_size=population_size)
    study = optuna.create_study(directions=["minimize"] * 2, sampler=sampler)
    study.optimize(objective, n_trials=population_size * 2)


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
        sampler = NSGAIISampler(population_size=2, constraints_func=constraints_func)

    study = optuna.create_study(directions=["minimize"] * n_objectives, sampler=sampler)
    study.optimize(
        lambda t: [t.suggest_float(f"x{i}", 0, 1) for i in range(n_objectives)], n_trials=n_trials
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
        sampler = NSGAIISampler(population_size=2, constraints_func=constraints_func)

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


@pytest.mark.parametrize("direction1", [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
@pytest.mark.parametrize("direction2", [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
@pytest.mark.parametrize(
    "constraints_list",
    [
        [[]],  # empty constraint
        [[-float("inf")], [-1], [0]],  # single constraint
        [
            [c1, c2] for c1 in [-float("inf"), -1, 0] for c2 in [-float("inf"), -1, 0]
        ],  # multiple constraints
    ],
)
def test_constrained_dominates_feasible_vs_feasible(
    direction1: StudyDirection, direction2: StudyDirection, constraints_list: list[list[float]]
) -> None:
    directions = [direction1, direction2]
    # Check all pairs of trials consisting of these values, i.e.,
    # [-inf, -inf], [-inf, -1], [-inf, 1], [-inf, inf], [-1, -inf], ...
    values_list = [
        [x, y]
        for x in [-float("inf"), -1, 1, float("inf")]
        for y in [-float("inf"), -1, 1, float("inf")]
    ]
    values_constraints_list = [(vs, cs) for vs in values_list for cs in constraints_list]

    # The results of _constrained_dominates match _dominates in all feasible cases.
    for values1, constraints1 in values_constraints_list:
        for values2, constraints2 in values_constraints_list:
            t1 = _create_frozen_trial(number=0, values=values1, constraints=constraints1)
            t2 = _create_frozen_trial(number=1, values=values2, constraints=constraints2)
            assert _constrained_dominates(t1, t2, directions) == _dominates(t1, t2, directions)


@pytest.mark.parametrize("direction", [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
def test_constrained_dominates_feasible_vs_infeasible(
    direction: StudyDirection,
) -> None:
    # Check all pairs of trials consisting of these constraint values.
    constraints_1d_feasible = [-float("inf"), -1, 0]
    constraints_1d_infeasible = [2, float("inf")]

    directions = [direction]

    # Feasible constraints.
    constraints_list1 = [
        [c1, c2] for c1 in constraints_1d_feasible for c2 in constraints_1d_feasible
    ]
    # Infeasible constraints.
    constraints_list2 = [
        [c1, c2]
        for c1 in constraints_1d_feasible + constraints_1d_infeasible
        for c2 in constraints_1d_infeasible
    ]

    # In the following code, we test that the feasible trials always dominate
    # the infeasible trials.
    for constraints1 in constraints_list1:
        for constraints2 in constraints_list2:
            t1 = _create_frozen_trial(number=0, values=[0], constraints=constraints1)
            t2 = _create_frozen_trial(number=1, values=[1], constraints=constraints2)
            assert _constrained_dominates(t1, t2, directions)
            assert not _constrained_dominates(t2, t1, directions)

            t1 = _create_frozen_trial(number=0, values=[1], constraints=constraints1)
            t2 = _create_frozen_trial(number=1, values=[0], constraints=constraints2)
            assert _constrained_dominates(t1, t2, directions)
            assert not _constrained_dominates(t2, t1, directions)


@pytest.mark.parametrize("direction", [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
def test_constrained_dominates_infeasible_vs_infeasible(direction: StudyDirection) -> None:
    inf = float("inf")
    directions = [direction]

    # The following table illustrates the violations of some constraint values.
    # When both trials are infeasible, the trial with smaller violation dominates
    # the one with larger violation.
    #
    #                       c2
    #      ╔═════╤═════╤═════╤═════╤═════╤═════╗
    #      ║     │ -1  │  0  │  1  │  2  │  ∞  ║
    #      ╟─────┼─────┼─────┼─────┼─────┼─────╢
    #      ║ -1  │           │  1  │  2  │  ∞  ║
    #      ╟─────┼─feasible ─┼─────┼─────┼─────╢
    #      ║  0  │           │  1  │  2  │  ∞  ║
    # c1   ╟─────┼─────┼─────┼─────┼─────┼─────╢
    #      ║  1  │  1  │  1  │  2  │  3  │  ∞  ║
    #      ╟─────┼─────┼─────┼─────┼─────┼─────╢
    #      ║  2  │  2  │  2  │  3  │  4  │  ∞  ║
    #      ╟─────┼─────┼─────┼─────┼─────┼─────╢
    #      ║  ∞  │  ∞  │  ∞  │  ∞  │  ∞  │  ∞  ║
    #      ╚═════╧═════╧═════╧═════╧═════╧═════╝
    #

    # Check all pairs of these constraints.
    constraints_infeasible_sorted: list[list[list[float]]]
    constraints_infeasible_sorted = [
        # These constraints have violation 1.
        [[1, -inf], [1, -1], [1, 0], [0, 1], [-1, 1], [-inf, 1]],
        # These constraints have violation 2.
        [[2, -inf], [2, -1], [2, 0], [1, 1], [0, 2], [-1, 2], [-inf, 2]],
        # These constraints have violation 3.
        [[3, -inf], [3, -1], [3, 0], [2, 1], [1, 2], [0, 3], [-1, 3], [-inf, 3]],
        # These constraints have violation inf.
        [
            [-inf, inf],
            [-1, inf],
            [0, inf],
            [1, inf],
            [inf, inf],
            [inf, 1],
            [inf, 0],
            [inf, -1],
            [inf, -inf],
        ],
    ]

    # Check that constraints with smaller violations dominate constraints with larger violation.
    for i in range(len(constraints_infeasible_sorted)):
        for j in range(i + 1, len(constraints_infeasible_sorted)):
            # Every constraint in constraints_infeasible_sorted[i] dominates
            # every constraint in constraints_infeasible_sorted[j].
            for constraints1 in constraints_infeasible_sorted[i]:
                for constraints2 in constraints_infeasible_sorted[j]:
                    t1 = _create_frozen_trial(number=0, values=[0], constraints=constraints1)
                    t2 = _create_frozen_trial(number=1, values=[1], constraints=constraints2)
                    assert _constrained_dominates(t1, t2, directions)
                    assert not _constrained_dominates(t2, t1, directions)

                    t1 = _create_frozen_trial(number=0, values=[1], constraints=constraints1)
                    t2 = _create_frozen_trial(number=1, values=[0], constraints=constraints2)
                    assert _constrained_dominates(t1, t2, directions)
                    assert not _constrained_dominates(t2, t1, directions)

    # Check that constraints with same violations are incomparable.
    for constraints_with_same_violations in constraints_infeasible_sorted:
        for constraints1 in constraints_with_same_violations:
            for constraints2 in constraints_with_same_violations:
                t1 = _create_frozen_trial(number=0, values=[0], constraints=constraints1)
                t2 = _create_frozen_trial(number=1, values=[1], constraints=constraints2)
                assert not _constrained_dominates(t1, t2, directions)
                assert not _constrained_dominates(t2, t1, directions)


def _assert_population_per_rank(
    trials: list[FrozenTrial],
    direction: list[StudyDirection],
    population_per_rank: list[list[FrozenTrial]],
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


@pytest.mark.parametrize("direction1", [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
@pytest.mark.parametrize("direction2", [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE])
def test_rank_population_no_constraints(
    direction1: StudyDirection, direction2: StudyDirection
) -> None:
    directions = [direction1, direction2]
    value_list = [10, 20, 20, 30, float("inf"), float("inf"), -float("inf")]
    values = [[v1, v2] for v1 in value_list for v2 in value_list]

    trials = [_create_frozen_trial(number=i, values=v) for i, v in enumerate(values)]
    population_per_rank = _rank_population(trials, directions)
    _assert_population_per_rank(trials, directions, population_per_rank)


def test_rank_population_with_constraints() -> None:
    value_list = [10, 20, 20, 30, float("inf"), float("inf"), -float("inf")]
    values = [[v1, v2] for v1 in value_list for v2 in value_list]

    constraint_list = [-float("inf"), -2, 0, 1, 2, 3, float("inf")]
    constraints = [[c1, c2] for c1 in constraint_list for c2 in constraint_list]

    trials = [
        _create_frozen_trial(number=i, values=v, constraints=c)
        for i, (v, c) in enumerate(itertools.product(values, constraints))
    ]
    directions = [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE]
    population_per_rank = _rank_population(trials, directions, is_constrained=True)
    _assert_population_per_rank(trials, directions, population_per_rank)


def test_validate_constraints() -> None:
    # Nan is not allowed in constraints.
    with pytest.raises(ValueError):
        _validate_constraints(
            [_create_frozen_trial(number=0, values=[1], constraints=[0, float("nan")])],
            is_constrained=True,
        )

    # Different numbers of constraints are not allowed.
    with pytest.raises(ValueError):
        _validate_constraints(
            [
                _create_frozen_trial(number=0, values=[1], constraints=[0]),
                _create_frozen_trial(number=1, values=[1], constraints=[0, 1]),
            ],
            is_constrained=True,
        )


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
def test_rank_population_missing_constraint_values(
    values_and_constraints: list[tuple[list[float], list[float]]],
) -> None:
    values_dim = len(values_and_constraints[0][0])
    for directions in itertools.product(
        [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE], repeat=values_dim
    ):
        trials = [
            _create_frozen_trial(number=i, values=v, constraints=c)
            for i, (v, c) in enumerate(values_and_constraints)
        ]

        with pytest.warns(UserWarning):
            _validate_constraints(trials, is_constrained=True)
        population_per_rank = _rank_population(trials, list(directions), is_constrained=True)
        _assert_population_per_rank(trials, list(directions), population_per_rank)


@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_rank_population_empty(n_dims: int) -> None:
    for directions in itertools.product(
        [StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE], repeat=n_dims
    ):
        trials: list[FrozenTrial] = []
        population_per_rank = _rank_population(trials, list(directions))
        assert population_per_rank == []


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
            [float("inf"), float("inf"), float("inf")],
        ),
        (
            [[float("-inf")], [1], [2]],
            [float("inf"), float("inf"), float("inf")],
        ),
        ([[float("inf")], [float("inf")], [float("inf")]], [0, 0, 0]),
        ([[-float("inf")], [-float("inf")], [-float("inf")]], [0, 0, 0]),
        ([[-float("inf")], [float("inf")]], [float("inf"), float("inf")]),
        (
            [[-float("inf")], [-float("inf")], [-float("inf")], [0], [1], [2], [float("inf")]],
            [0, 0, float("inf"), float("inf"), 1, float("inf"), float("inf")],
        ),
    ],
)
def test_calc_crowding_distance(values: list[list[float]], expected_dist: list[float]) -> None:
    trials = [_create_frozen_trial(number=i, values=value) for i, value in enumerate(values)]
    crowding_dist = _calc_crowding_distance(trials)
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
def test_crowding_distance_sort(values: list[list[float]]) -> None:
    """Checks that trials are sorted by the values of `_calc_crowding_distance`."""
    trials = [_create_frozen_trial(number=i, values=value) for i, value in enumerate(values)]
    crowding_dist = _calc_crowding_distance(trials)
    _crowding_distance_sort(trials)
    sorted_dist = [crowding_dist[t.number] for t in trials]
    assert sorted_dist == sorted(sorted_dist, reverse=True)


def test_constraints_func_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        NSGAIISampler(constraints_func=lambda _: [0])


def test_elite_population_selection_strategy_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        NSGAIISampler(elite_population_selection_strategy=lambda study, population: [])


def test_child_generation_strategy_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        NSGAIISampler(child_generation_strategy=lambda study, search_space, parent_population: {})


def test_after_trial_strategy_experimental_warning() -> None:
    with pytest.warns(optuna.exceptions.ExperimentalWarning):
        NSGAIISampler(after_trial_strategy=lambda study, trial, state, value: None)


def test_elite_population_selection_strategy_invalid_value() -> None:
    with pytest.raises(ValueError):
        NSGAIIElitePopulationSelectionStrategy(population_size=1)


@pytest.mark.parametrize(
    "objectives, expected_elite_population",
    [
        (
            [[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]],
            [[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]],
        ),
        (
            [[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 4.0]],
            [[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 4.0]],
        ),
        (
            [[1.0, 2.0], [2.0, 1.0], [5.0, 3.0], [3.0, 5.0], [4.0, 4.0]],
            [[1.0, 2.0], [2.0, 1.0], [5.0, 3.0], [3.0, 5.0]],
        ),
    ],
)
def test_elite_population_selection_strategy_result(
    objectives: list[list[float]],
    expected_elite_population: list[list[float]],
) -> None:
    population_size = 4
    elite_population_selection_strategy = NSGAIIElitePopulationSelectionStrategy(
        population_size=population_size
    )
    study = optuna.create_study(directions=["minimize", "minimize"])
    study.add_trials([optuna.create_trial(values=values) for values in objectives])
    elite_population_values = [
        trial.values for trial in elite_population_selection_strategy(study, study.get_trials())
    ]
    assert len(elite_population_values) == population_size
    for values in elite_population_values:
        assert values in expected_elite_population


@pytest.mark.parametrize(
    "mutation_prob,crossover,crossover_prob,swapping_prob",
    [
        (1.2, UniformCrossover(), 0.9, 0.5),
        (-0.2, UniformCrossover(), 0.9, 0.5),
        (None, UniformCrossover(), 1.2, 0.5),
        (None, UniformCrossover(), -0.2, 0.5),
        (None, UniformCrossover(), 0.9, 1.2),
        (None, UniformCrossover(), 0.9, -0.2),
        (None, 3, 0.9, 0.5),
    ],
)
def test_child_generation_strategy_invalid_value(
    mutation_prob: float,
    crossover: BaseCrossover | int,
    crossover_prob: float,
    swapping_prob: float,
) -> None:
    with pytest.raises(ValueError):
        NSGAIIChildGenerationStrategy(
            mutation_prob=mutation_prob,
            crossover=crossover,  # type: ignore[arg-type]
            crossover_prob=crossover_prob,
            swapping_prob=swapping_prob,
            rng=LazyRandomState(),
        )


@pytest.mark.parametrize(
    "mutation_prob,child_params",
    [(0.0, {"x": 1.0, "y": 0.0}), (1.0, {})],
)
def test_child_generation_strategy_mutation_prob(
    mutation_prob: int, child_params: dict[str, float]
) -> None:
    child_generation_strategy = NSGAIIChildGenerationStrategy(
        crossover_prob=0.0,
        crossover=UniformCrossover(),
        mutation_prob=mutation_prob,
        swapping_prob=0.5,
        rng=LazyRandomState(seed=1),
    )
    study = MagicMock(spec=optuna.study.Study)
    search_space = MagicMock(spec=dict)
    search_space.keys.return_value = ["x", "y"]
    parent_population = [
        optuna.trial.create_trial(
            params={"x": 1.0, "y": 0},
            distributions={
                "x": FloatDistribution(0, 10),
                "y": CategoricalDistribution([-1, 0, 1]),
            },
            value=5.0,
        )
    ]
    assert child_generation_strategy(study, search_space, parent_population) == child_params


def test_child_generation_strategy_generation_key() -> None:
    n_params = 2

    def objective(trial: optuna.Trial) -> list[float]:
        xs = [trial.suggest_float(f"x{dim}", -10, 10) for dim in range(n_params)]
        return xs

    mock_func = MagicMock(spec=Callable, return_value={"x0": 0.0, "x1": 1.1})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        study = optuna.create_study(
            sampler=NSGAIISampler(population_size=2, child_generation_strategy=mock_func),
            directions=["minimize", "minimize"],
        )
    study.optimize(objective, n_trials=3)
    assert mock_func.call_count == 1
    for i, trial in enumerate(study.get_trials()):
        if i < 2:
            assert trial.system_attrs[NSGAIISampler._GENERATION_KEY] == 0
        elif i == 2:
            assert trial.system_attrs[NSGAIISampler._GENERATION_KEY] == 1


@patch(
    "optuna.samplers.nsgaii._child_generation_strategy.perform_crossover",
    return_value={"x": 3.0, "y": 2.0},
)
def test_child_generation_strategy_crossover_prob(mock_func: MagicMock) -> None:
    study = MagicMock(spec=optuna.study.Study)
    search_space = MagicMock(spec=dict)
    search_space.keys.return_value = ["x", "y"]
    parent_population = [
        optuna.trial.create_trial(
            params={"x": 1.0, "y": 0},
            distributions={
                "x": FloatDistribution(0, 10),
                "y": CategoricalDistribution([-1, 0, 1]),
            },
            value=5.0,
        )
    ]
    child_generation_strategy_always_not_crossover = NSGAIIChildGenerationStrategy(
        crossover_prob=0.0,
        crossover=UniformCrossover(),
        mutation_prob=None,
        swapping_prob=0.5,
        rng=LazyRandomState(seed=1),
    )
    assert child_generation_strategy_always_not_crossover(
        study, search_space, parent_population
    ) == {"x": 1.0}
    assert mock_func.call_count == 0

    child_generation_strategy_always_crossover = NSGAIIChildGenerationStrategy(
        crossover_prob=1.0,
        crossover=UniformCrossover(),
        mutation_prob=0.0,
        swapping_prob=0.5,
        rng=LazyRandomState(),
    )
    assert child_generation_strategy_always_crossover(study, search_space, parent_population) == {
        "x": 3.0,
        "y": 2.0,
    }
    assert mock_func.call_count == 1


def test_call_after_trial_of_random_sampler() -> None:
    sampler = NSGAIISampler()
    study = optuna.create_study(sampler=sampler)
    with patch.object(
        sampler._random_sampler, "after_trial", wraps=sampler._random_sampler.after_trial
    ) as mock_object:
        study.optimize(lambda _: 1.0, n_trials=1)
        assert mock_object.call_count == 1


def test_call_after_trial_of_after_trial_strategy() -> None:
    sampler = NSGAIISampler()
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
    search_space: dict[str, BaseDistribution] = {
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
    search_space: dict[str, BaseDistribution] = {
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
        return np.full(kwargs.get("size"), rand_value)  # type: ignore[arg-type]

    rng = Mock()
    rng.rand = Mock(side_effect=_rand)
    rng.normal = Mock(side_effect=_normal)
    child_params = crossover.crossover(parent_params, rng, study, numerical_transform.bounds)
    np.testing.assert_almost_equal(child_params, expected_params)
