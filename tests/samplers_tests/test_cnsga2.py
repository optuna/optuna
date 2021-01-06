from typing import List

import optuna
from optuna.samplers import CNSGAIISampler
from optuna.samplers._cnsga2 import _CONSTRAINTS_KEY
from optuna.study import StudyDirection


def test_dominates_feasible() -> None:
    # Single objective.
    directions = [StudyDirection.MINIMIZE]
    trials = [
        _create_frozen_trial(0, [10], [0]),
        _create_frozen_trial(1, [20], [0]),
        _create_frozen_trial(2, [20], [0]),
        _create_frozen_trial(3, [30], [0]),
    ]
    population_per_rank = CNSGAIISampler._fast_non_dominated_sort(trials, directions)
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
    population_per_rank = CNSGAIISampler._fast_non_dominated_sort(trials, directions)
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
    population_per_rank = CNSGAIISampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {2},
        {0, 3, 5},
        {1},
        {4},
    ]


def test_dominates_infeasible() -> None:
    # Single objective.
    directions = [StudyDirection.MINIMIZE]
    trials = [
        _create_frozen_trial(0, [10], [1]),
        _create_frozen_trial(1, [20], [3]),
        _create_frozen_trial(2, [20], [2]),
        _create_frozen_trial(3, [30], [1]),
    ]
    population_per_rank = CNSGAIISampler._fast_non_dominated_sort(trials, directions)
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
    population_per_rank = CNSGAIISampler._fast_non_dominated_sort(trials, directions)
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
    population_per_rank = CNSGAIISampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {0, 1},
        {3, 4, 5},
        {2},
    ]


def test_dominates_feasible_infeasible() -> None:
    # Single objective.
    directions = [StudyDirection.MINIMIZE]
    trials = [
        _create_frozen_trial(0, [10], [0]),
        _create_frozen_trial(1, [20], [-1]),
        _create_frozen_trial(2, [20], [-2]),
        _create_frozen_trial(3, [30], [1]),
    ]
    population_per_rank = CNSGAIISampler._fast_non_dominated_sort(trials, directions)
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
    population_per_rank = CNSGAIISampler._fast_non_dominated_sort(trials, directions)
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
    population_per_rank = CNSGAIISampler._fast_non_dominated_sort(trials, directions)
    assert [{t.number for t in population} for population in population_per_rank] == [
        {0, 3, 5},
        {4},
        {2},
        {1},
    ]


def _create_frozen_trial(
    number: int, values: List[float], constraints: List[float]
) -> optuna.trial.FrozenTrial:
    return optuna.trial.FrozenTrial(
        number=number,
        trial_id=number,
        state=optuna.trial.TrialState.COMPLETE,
        value=None,
        datetime_start=None,
        datetime_complete=None,
        params={},
        distributions={},
        user_attrs={},
        system_attrs={_CONSTRAINTS_KEY: constraints},
        intermediate_values={},
        values=values,
    )
