from collections import defaultdict
from typing import List
from typing import Tuple

import numpy as np
import pytest

from optuna.samplers.nsgaiii import _associate
from optuna.samplers.nsgaiii import _niching
from optuna.samplers.nsgaiii import _normalize
from optuna.samplers.nsgaiii import generate_default_reference_point
from optuna.trial import create_trial


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
def test_reference_point(dims_and_dividing_parameter: Tuple[int, int, List[List[int]]]) -> None:
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
        target_population_size, population, reference_points_per_count, ref2pops, seed=0
    )
    expected_additional_elite_population = [population[3], population[1]]
    assert actual_additional_elite_population == expected_additional_elite_population
