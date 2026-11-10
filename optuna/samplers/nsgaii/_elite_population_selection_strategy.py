from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from optuna.samplers.nsgaii._constraints_evaluation import _evaluate_penalty
from optuna.samplers.nsgaii._constraints_evaluation import _validate_constraints
from optuna.study import StudyDirection
from optuna.study._constrained_optimization import _is_constrained_optimization
from optuna.study._multi_objective import _fast_non_domination_rank


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from optuna.study import Study
    from optuna.trial import FrozenTrial


class NSGAIIElitePopulationSelectionStrategy:
    def __init__(
        self,
        *,
        population_size: int,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("`population_size` must be greater than or equal to 2.")

        self._population_size = population_size
        self._constraints_func = constraints_func

    def __call__(self, study: Study, population: list[FrozenTrial]) -> list[FrozenTrial]:
        """Select elite population from the given trials by NSGA-II algorithm.

        Args:
            study:
                Target study object.
            population:
                Trials in the study.

        Returns:
            A list of trials that are selected as elite population.
        """
        is_constrained = _is_constrained_optimization(population)
        _validate_constraints(population, is_constrained=is_constrained)
        population_per_rank = _rank_population(
            population, study.directions, is_constrained=is_constrained
        )
        elite_population: list[FrozenTrial] = []
        for individuals in population_per_rank:
            if len(elite_population) + len(individuals) < self._population_size:
                elite_population.extend(individuals)
            else:
                n = self._population_size - len(elite_population)
                _crowding_distance_sort(individuals)
                elite_population.extend(individuals[:n])
                break

        return elite_population


def _calc_crowding_distance(population: list[FrozenTrial]) -> defaultdict[int, float]:
    """Calculates the crowding distance of population.

    We define the crowding distance as the summation of the crowding distance of each dimension
    of value calculated as follows:

    * If all values in that dimension are the same, i.e., [1, 1, 1] or [inf, inf],
      the crowding distances of all trials in that dimension are zero.
    * Otherwise, the crowding distances of that dimension is the difference between
      two nearest values besides that value, one above and one below, divided by the difference
      between the maximal and minimal finite value of that dimension. Please note that:
        * the nearest value below the minimum is considered to be -inf and the
          nearest value above the maximum is considered to be inf, and
        * inf - inf and (-inf) - (-inf) is considered to be zero.
    """

    manhattan_distances: defaultdict[int, float] = defaultdict(float)
    n_pop = len(population)
    if n_pop == 0:
        return manhattan_distances

    values = np.array([trial.values for trial in population], dtype=np.float64)
    distances = np.zeros(n_pop, dtype=np.float64)
    order = np.arange(n_pop)
    padded = np.empty(n_pop + 2, dtype=np.float64)
    padded[0] = -np.inf
    padded[-1] = np.inf
    has_non_constant_dim = False

    for i in range(values.shape[1]):
        col = values[order, i]
        order = order[np.argsort(col, kind="mergesort")]
        sorted_col = values[order, i]

        # If all trials in population have the same value in the i-th dimension, ignore the
        # objective dimension since it does not make difference.
        if sorted_col[0] == sorted_col[-1]:
            continue

        has_non_constant_dim = True
        low_idx = int(np.searchsorted(sorted_col, -np.inf, side="right"))
        high_idx = int(np.searchsorted(sorted_col, np.inf, side="left")) - 1
        v_min = float(sorted_col[low_idx]) if low_idx < n_pop else float("inf")
        v_max = float(sorted_col[high_idx]) if high_idx >= 0 else -float("inf")

        width = v_max - v_min
        if width <= 0:
            # width == 0 or width == -inf
            width = 1.0

        padded[1:-1] = sorted_col
        left = padded[:-2]
        right = padded[2:]
        with np.errstate(invalid="ignore"):
            # inf - inf and (-inf) - (-inf) is considered to be zero.
            gaps = np.where(left == right, 0.0, right - left)
        distances[order] += gaps / width

    if has_non_constant_dim:
        for trial, dist in zip(population, distances.tolist()):
            manhattan_distances[trial.number] = dist
    population[:] = [population[idx] for idx in order.tolist()]
    return manhattan_distances


def _crowding_distance_sort(population: list[FrozenTrial]) -> None:
    manhattan_distances = _calc_crowding_distance(population)
    population.sort(key=lambda x: manhattan_distances[x.number])
    population.reverse()


def _rank_population(
    population: list[FrozenTrial],
    directions: Sequence[StudyDirection],
    *,
    is_constrained: bool = False,
) -> list[list[FrozenTrial]]:
    if len(population) == 0:
        return []

    objective_values = np.array([trial.values for trial in population], dtype=np.float64)
    objective_values *= np.array(
        [-1.0 if d == StudyDirection.MAXIMIZE else 1.0 for d in directions]
    )
    penalty = _evaluate_penalty(population) if is_constrained else None

    domination_ranks = _fast_non_domination_rank(objective_values, penalty=penalty)
    population_per_rank: list[list[FrozenTrial]] = [[] for _ in range(max(domination_ranks) + 1)]
    for trial, rank in zip(population, domination_ranks):
        if rank == -1:
            continue
        population_per_rank[rank].append(trial)

    return population_per_rank
