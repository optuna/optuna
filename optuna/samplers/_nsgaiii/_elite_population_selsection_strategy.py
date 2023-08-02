from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Sequence
import itertools
import math

import numpy as np

from optuna.samplers.nsgaii._dominates import _constrained_dominates
from optuna.samplers.nsgaii._dominates import _validate_constraints
from optuna.samplers.nsgaii._elite_population_selection_strategy import _fast_non_dominated_sort
from optuna.study import Study
from optuna.study._multi_objective import _dominates
from optuna.trial import FrozenTrial


# Define a coefficient for scaling intervals, used in _filter_inf() to replace +-inf.
_COEF = 3


class NSGAIIIElitePopulationSelectionStrategy:
    def __init__(
        self,
        *,
        population_size: int,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        reference_points: np.ndarray | None = None,
        dividing_parameter: int = 3,
        seed: int | None = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("`population_size` must be greater than or equal to 2.")

        self._population_size = population_size
        self._constraints_func = constraints_func
        self._reference_points = reference_points
        self._dividing_parameter = dividing_parameter
        self._rng = np.random.RandomState(seed)

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
        _validate_constraints(population, self._constraints_func)
        dominates = _dominates if self._constraints_func is None else _constrained_dominates
        population_per_rank = _fast_non_dominated_sort(population, study.directions, dominates)

        elite_population: list[FrozenTrial] = []
        for population in population_per_rank:
            if len(elite_population) + len(population) < self._population_size:
                elite_population.extend(population)
            else:
                n_objectives = len(study.directions)
                # Construct reference points in the first run.
                if self._reference_points is None:
                    self._reference_points = _generate_default_reference_point(
                        n_objectives, self._dividing_parameter
                    )
                elif np.shape(self._reference_points)[1] != n_objectives:
                    raise ValueError(
                        "The dimension of reference points vectors must be the same as the number "
                        "of objectives of the study."
                    )

                # Normalize objective values after filtering +-inf.
                objective_matrix = _normalize_objective_values(
                    _filter_inf(elite_population + population)
                )
                (
                    closest_reference_points,
                    distance_reference_points,
                ) = _associate_individuals_with_reference_points(
                    objective_matrix, self._reference_points
                )

                elite_population_num = len(elite_population)
                target_population_size = self._population_size - elite_population_num
                additional_elite_population = _preserve_niche_individuals(
                    target_population_size,
                    elite_population_num,
                    population,
                    closest_reference_points,
                    distance_reference_points,
                    self._rng,
                )
                elite_population.extend(additional_elite_population)
                break
        return elite_population


# TODO(Shinichi) Replace with math.comb after support for python3.7 is deprecated.
# This function calculates n multi-choose k, which is the total number of combinations with
# repetition of size k from n items. This is equally re-written as math.comb(n+k-1, k)
def _multi_choose(n: int, k: int) -> int:
    return math.factorial(n + k - 1) // math.factorial(k) // math.factorial(n - 1)


def _generate_default_reference_point(
    n_objectives: int, dividing_parameter: int = 3
) -> np.ndarray:
    """Generates default reference points which are `uniformly` spread on a hyperplane."""
    reference_points = np.zeros(
        (
            _multi_choose(n_objectives, dividing_parameter),
            n_objectives,
        )
    )
    for i, comb in enumerate(
        itertools.combinations_with_replacement(range(n_objectives), dividing_parameter)
    ):
        for j in comb:
            reference_points[i, j] += 1.0
    return reference_points


def _filter_inf(population: list[FrozenTrial]) -> np.ndarray:
    # Collect all objective values.
    n_objectives = len(population[0].values)
    objective_matrix = np.zeros((len(population), n_objectives))
    for i, trial in enumerate(population):
        objective_matrix[i] = np.array(trial.values, dtype=float)

    mask_posinf = np.isposinf(objective_matrix)
    mask_neginf = np.isneginf(objective_matrix)

    # Replace +-inf with nan temporary to get max and min.
    objective_matrix[mask_posinf + mask_neginf] = np.nan
    nadir_point = np.nanmax(objective_matrix, axis=0)
    ideal_point = np.nanmin(objective_matrix, axis=0)
    interval = nadir_point - ideal_point

    # TODO(Shinichi) reconsider alternative value for inf.
    rows_posinf, cols_posinf = np.where(mask_posinf)
    objective_matrix[rows_posinf, cols_posinf] = (
        nadir_point[cols_posinf] + _COEF * interval[cols_posinf]
    )
    rows_neginf, cols_neginf = np.where(mask_neginf)
    objective_matrix[rows_neginf, cols_neginf] = (
        ideal_point[cols_neginf] - _COEF * interval[cols_neginf]
    )

    return objective_matrix


def _normalize_objective_values(objective_matrix: np.ndarray) -> np.ndarray:
    """Normalizes objective values of population.

    An ideal point z* consists of minimums in each axis. Each objective value of population is
    then subtracted by the ideal point.
    An extreme point of each axis is (originally) defined as a minimum solution of achievement
    scalarizing function from the population. After that, intercepts are calculate as intercepts
    of hyperplane which has all the extreme points on it and used to rescale objective values.

    We adopt weights and achievement scalarizing function(ASF) used in pre-print of the NSGA-III
    paper (See https://www.egr.msu.edu/~kdeb/papers/k2012009.pdf).
    """
    n_objectives = np.shape(objective_matrix)[1]
    # Subtract ideal point from objective values.
    objective_matrix -= np.min(objective_matrix, axis=0)
    # Initialize weights.
    weights = np.eye(n_objectives)
    weights[weights == 0] = 1e6

    # Calculate extreme points to normalize objective values.
    # TODO(Shinichi) Reimplement to reduce time complexity.
    asf_value = np.max(
        np.einsum("nm,dm->dnm", objective_matrix, weights),
        axis=2,
    )
    extreme_points = objective_matrix[np.argmin(asf_value, axis=1), :]

    # Normalize objective_matrix with extreme points.
    # Note that extreme_points can be degenerate, but no proper operation is remarked in the
    # paper. Therefore, the maximum value of population in each axis is used in such cases.
    if np.all(np.isfinite(extreme_points)) and np.linalg.matrix_rank(extreme_points) == len(
        extreme_points
    ):
        intercepts_inv = np.linalg.solve(extreme_points, np.ones(n_objectives))
    else:
        intercepts_inv = 1 / np.max(objective_matrix, axis=0)
    objective_matrix *= np.where(np.isfinite(intercepts_inv), intercepts_inv, 1)

    return objective_matrix


def _associate_individuals_with_reference_points(
    objective_matrix: np.ndarray, reference_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Associates each objective value to the closest reference point.

    Associate each normalized objective value to the closest reference point. The distance is
    calculated by Euclidean norm.

    Args:
        objective_matrix:
            A 2 dimension ``numpy.ndarray`` with columns of objective dimension and rows of
            generation size. Each row is the normalized objective value of the corresponding
            individual.

    Returns:
        closest_reference_points:
            A ``numpy.ndarray`` with rows of generation size. Each row is the index of
            the closest reference point to the corresponding individual.
        distance_reference_points:
            A ``numpy.ndarray`` with rows of generation size. Each row is the distance from
            the corresponding individual to the closest reference point.
    """
    # TODO(Shinichi) Implement faster assignment for the default reference points because it does
    # not seem necessary to calculate distance from all reference points.

    # TODO(Shinichi) Normalize reference_points in constructor to remove reference_point_norms.
    # In addition, the minimum distance from each reference point can be replaced with maximum
    # inner product between the given individual and each normalized reference points.

    # distance_from_reference_lines is a ndarray of shape (n, p), where n is the size of the
    # population and p is the number of reference points. Its (i,j) entry keeps distance between
    # the i-th individual values and the j-th reference line.
    reference_point_norm_squared = np.linalg.norm(reference_points, axis=1) ** 2
    perpendicular_vectors_to_reference_lines = np.einsum(
        "ni,pi,p,pm->npm",
        objective_matrix,
        reference_points,
        1 / reference_point_norm_squared,
        reference_points,
    )
    distance_from_reference_lines = np.linalg.norm(
        objective_matrix[:, np.newaxis, :] - perpendicular_vectors_to_reference_lines,
        axis=2,
    )
    closest_reference_points: np.ndarray = np.argmin(distance_from_reference_lines, axis=1)
    distance_reference_points: np.ndarray = np.min(distance_from_reference_lines, axis=1)

    return closest_reference_points, distance_reference_points


def _preserve_niche_individuals(
    target_population_size: int,
    elite_population_num: int,
    population: list[FrozenTrial],
    closest_reference_points: np.ndarray,
    distance_reference_points: np.ndarray,
    rng: np.random.RandomState,
) -> list[FrozenTrial]:
    """Determine who survives form the borderline front.

    Who survive form the borderline front is determined according to the sparsity of each closest
    reference point. The algorithm picks a reference point from those who have the least neighbors
    in elite population and adds one of borderline front member who has the same closest reference
    point.

    Args:
        target_population_size:
            The number of individuals to select.
        elite_population_num:
            The number of individuals which are already selected as the elite population.
        population:
            List of all the trials in the current surviving generation.
        distance_reference_points:
            A ``numpy.ndarray`` with rows of generation size. Each row is the distance from the
            corresponding individual to the closest reference point.
        closest_reference_points:
            A ``numpy.ndarray`` with rows of generation size. Each row is the index of the closest
            reference point to the corresponding individual.
        rng:
            Random number generator.

    Returns:
        A list of trials which are selected as the next generation.
    """
    if len(population) < target_population_size:
        raise ValueError(
            "The population size must be greater than or equal to the target population size."
        )

    # reference_point_to_borderline_population keeps pairs of a neighbor and the distance of
    # each reference point from borderline front population.
    reference_point_to_borderline_population = defaultdict(list)
    for i, reference_point_idx in enumerate(closest_reference_points[elite_population_num:]):
        population_idx = i + elite_population_num
        reference_point_to_borderline_population[reference_point_idx].append(
            (distance_reference_points[population_idx], i)
        )

    # reference_points_to_elite_population_count keeps how many elite neighbors each reference
    # point has.
    reference_point_to_elite_population_count: dict[int, int] = defaultdict(int)
    for i, reference_point_idx in enumerate(closest_reference_points[:elite_population_num]):
        reference_point_to_elite_population_count[reference_point_idx] += 1
    # nearest_points_count_to_reference_points classifies reference points which have at least one
    # closest borderline population member by the number of elite neighbors they have.  Each key
    # corresponds to the number of elite neighbors and the value to the reference point indices.
    nearest_points_count_to_reference_points = defaultdict(list)
    for reference_point_idx in reference_point_to_borderline_population:
        elite_population_count = reference_point_to_elite_population_count[reference_point_idx]
        nearest_points_count_to_reference_points[elite_population_count].append(
            reference_point_idx
        )

    count = -1
    additional_elite_population: list[FrozenTrial] = []
    is_shuffled: defaultdict[int, bool] = defaultdict(bool)
    while len(additional_elite_population) < target_population_size:
        if len(nearest_points_count_to_reference_points[count]) == 0:
            count += 1
            rng.shuffle(nearest_points_count_to_reference_points[count])
            continue

        reference_point_idx = nearest_points_count_to_reference_points[count].pop()
        if count > 0 and not is_shuffled[reference_point_idx]:
            rng.shuffle(reference_point_to_borderline_population[reference_point_idx])
            is_shuffled[reference_point_idx] = True
        elif count == 0:
            reference_point_to_borderline_population[reference_point_idx].sort(reverse=True)

        _, selected_individual_id = reference_point_to_borderline_population[
            reference_point_idx
        ].pop()
        additional_elite_population.append(population[selected_individual_id])
        if reference_point_to_borderline_population[reference_point_idx]:
            nearest_points_count_to_reference_points[count + 1].append(reference_point_idx)

    return additional_elite_population
