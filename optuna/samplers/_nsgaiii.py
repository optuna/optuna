from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Sequence
import hashlib
import itertools
import math
from typing import Any

import numpy as np

import optuna
from optuna._experimental import experimental_class
from optuna.distributions import BaseDistribution
from optuna.samplers._base import BaseSampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers._random import RandomSampler
from optuna.samplers.nsgaii._after_trial_strategy import NSGAIIAfterTrialStrategy
from optuna.samplers.nsgaii._child_generation_strategy import NSGAIIChildGenerationStrategy
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.samplers.nsgaii._crossovers._uniform import UniformCrossover
from optuna.samplers.nsgaii._dominates import _constrained_dominates
from optuna.samplers.nsgaii._dominates import _validate_constraints
from optuna.samplers.nsgaii._elite_population_selection_strategy import _fast_non_dominated_sort
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.study._multi_objective import _dominates
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


# Define key names of `Trial.system_attrs`.
_GENERATION_KEY = "nsga3:generation"
_POPULATION_CACHE_KEY_PREFIX = "nsga3:population"

# Define a coefficient for scaling intervals, used in _filter_inf() to replace +-inf.
_COEF = 3


@experimental_class("3.2.0")
class NSGAIIISampler(BaseSampler):
    """Multi-objective sampler using the NSGA-III algorithm.

    NSGA-III stands for "Nondominated Sorting Genetic Algorithm III",
    which is a modified version of NSGA-II for many objective optimization problem.

    For further information about NSGA-III, please refer to the following papers:

    - `An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based
      Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints
      <https://doi.org/10.1109/TEVC.2013.2281535>`_
    - `An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based
      Nondominated Sorting Approach, Part II: Handling Constraints and Extending to an Adaptive
      Approach
      <https://doi.org/10.1109/TEVC.2013.2281534>`_

    Args:
        reference_points:
            A 2 dimension ``numpy.ndarray`` with objective dimension columns. Represents
            a list of reference points which is used to determine who to survive.
            After non-dominated sort, who out of borderline front are going to survived is
            determined according to how sparse the closest reference point of each individual is.
            In the default setting the algorithm uses `uniformly` spread points to diversify the
            result. It is also possible to reflect your `preferences` by giving an arbitrary set of
            `target` points since the algorithm prioritizes individuals around reference points.

        dividing_parameter:
            A parameter to determine the density of default reference points. This parameter
            determines how many divisions are made between reference points on each axis. The
            smaller this value is, the less reference points you have. The default value is 3.
            Note that this parameter is not used when ``reference_points`` is not :obj:`None`.

    .. note::
        Other parameters than ``reference_points`` and ``dividing_parameter`` are the same as
        :class:`~optuna.samplers.NSGAIISampler`.

    """

    def __init__(
        self,
        *,
        population_size: int = 50,
        mutation_prob: float | None = None,
        crossover: BaseCrossover | None = None,
        crossover_prob: float = 0.9,
        swapping_prob: float = 0.5,
        seed: int | None = None,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        reference_points: np.ndarray | None = None,
        dividing_parameter: int = 3,
        child_generation_strategy: Callable[
            [Study, dict[str, BaseDistribution], list[FrozenTrial]], dict[str, Any]
        ]
        | None = None,
        after_trial_strategy: Callable[
            [Study, FrozenTrial, TrialState, Sequence[float] | None], None
        ]
        | None = None,
    ) -> None:
        # TODO(ohta): Reconsider the default value of each parameter.

        if population_size < 2:
            raise ValueError("`population_size` must be greater than or equal to 2.")

        if crossover is None:
            crossover = UniformCrossover(swapping_prob)

        if not isinstance(crossover, BaseCrossover):
            raise ValueError(
                f"'{crossover}' is not a valid crossover."
                " For valid crossovers see"
                " https://optuna.readthedocs.io/en/stable/reference/samplers.html."
            )

        if population_size < crossover.n_parents:
            raise ValueError(
                f"Using {crossover},"
                f" the population size should be greater than or equal to {crossover.n_parents}."
                f" The specified `population_size` is {population_size}."
            )

        self._population_size = population_size
        self._random_sampler = RandomSampler(seed=seed)
        self._rng = LazyRandomState(seed)
        self._constraints_func = constraints_func
        self._reference_points = reference_points
        self._dividing_parameter = dividing_parameter
        self._search_space = IntersectionSearchSpace()
        self._child_generation_strategy = (
            child_generation_strategy
            or NSGAIIChildGenerationStrategy(
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob,
                swapping_prob=swapping_prob,
                crossover=crossover,
                constraints_func=constraints_func,
                seed=seed,
            )
        )
        self._after_trial_strategy = after_trial_strategy or NSGAIIAfterTrialStrategy(
            constraints_func=constraints_func
        )

    def reseed_rng(self) -> None:
        self._random_sampler.reseed_rng()
        self._rng.rng.seed()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # The `untransform` method of `optuna._transform._SearchSpaceTransform`
                # does not assume a single value,
                # so single value objects are not sampled with the `sample_relative` method,
                # but with the `sample_independent` method.
                continue
            search_space[name] = distribution
        return search_space

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        parent_generation, parent_population = self._collect_parent_population(study)

        generation = parent_generation + 1
        study._storage.set_trial_system_attr(trial._trial_id, _GENERATION_KEY, generation)

        if parent_generation < 0:
            return {}

        return self._child_generation_strategy(study, search_space, parent_population)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        # Following parameters are randomly sampled here.
        # 1. A parameter in the initial population/first generation.
        # 2. A parameter to mutate.
        # 3. A parameter excluded from the intersection search space.

        return self._random_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def _collect_parent_population(self, study: Study) -> tuple[int, list[FrozenTrial]]:
        trials = study.get_trials(deepcopy=False)

        generation_to_runnings = defaultdict(list)
        generation_to_population = defaultdict(list)
        for trial in trials:
            if _GENERATION_KEY not in trial.system_attrs:
                continue

            generation = trial.system_attrs[_GENERATION_KEY]
            if trial.state != optuna.trial.TrialState.COMPLETE:
                if trial.state == optuna.trial.TrialState.RUNNING:
                    generation_to_runnings[generation].append(trial)
                continue

            # Do not use trials whose states are not COMPLETE, or `constraint` will be unavailable.
            generation_to_population[generation].append(trial)

        hasher = hashlib.sha256()
        parent_population: list[FrozenTrial] = []
        parent_generation = -1
        while True:
            generation = parent_generation + 1
            population = generation_to_population[generation]

            # Under multi-worker settings, the population size might become larger than
            # `self._population_size`.
            if len(population) < self._population_size:
                break

            # [NOTE]
            # It's generally safe to assume that once the above condition is satisfied,
            # there are no additional individuals added to the generation (i.e., the members of
            # the generation have been fixed).
            # If the number of parallel workers is huge, this assumption can be broken, but
            # this is a very rare case and doesn't significantly impact optimization performance.
            # So we can ignore the case.

            # The cache key is calculated based on the key of the previous generation and
            # the remaining running trials in the current population.
            # If there are no running trials, the new cache key becomes exactly the same as
            # the previous one, and the cached content will be overwritten. This allows us to
            # skip redundant cache key calculations when this method is called for the subsequent
            # trials.
            for trial in generation_to_runnings[generation]:
                hasher.update(bytes(str(trial.number), "utf-8"))

            cache_key = "{}:{}".format(_POPULATION_CACHE_KEY_PREFIX, hasher.hexdigest())
            study_system_attrs = study._storage.get_study_system_attrs(study._study_id)
            cached_generation, cached_population_numbers = study_system_attrs.get(
                cache_key, (-1, [])
            )
            if cached_generation >= generation:
                generation = cached_generation
                population = [trials[n] for n in cached_population_numbers]
            else:
                population.extend(parent_population)
                population = self._select_elite_population(study, population)

                # To reduce the number of system attribute entries,
                # we cache the population information only if there are no running trials
                # (i.e., the information of the population has been fixed).
                # Usually, if there are no too delayed running trials, the single entry
                # will be used.
                if len(generation_to_runnings[generation]) == 0:
                    population_numbers = [t.number for t in population]
                    study._storage.set_study_system_attr(
                        study._study_id, cache_key, (generation, population_numbers)
                    )

            parent_generation = generation
            parent_population = population

        return parent_generation, parent_population

    def _select_elite_population(
        self, study: Study, population: list[FrozenTrial]
    ) -> list[FrozenTrial]:
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
                    self._rng.rng,
                )
                elite_population.extend(additional_elite_population)
                break
        return elite_population

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._random_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        assert state in [TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED]
        self._after_trial_strategy(study, trial, state, values)
        self._random_sampler.after_trial(study, trial, state, values)


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
