from collections import defaultdict
import hashlib
import itertools
import math
from typing import Any
from typing import Callable
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import warnings

import numpy as np

import optuna
from optuna.distributions import BaseDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.samplers._base import _process_constraints_after_trial
from optuna.samplers._base import BaseSampler
from optuna.samplers._random import RandomSampler
from optuna.samplers._search_space import IntersectionSearchSpace
from optuna.samplers.nsgaii._crossover import perform_crossover
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.samplers.nsgaii._crossovers._uniform import UniformCrossover
from optuna.samplers.nsgaii._sampler import _constrained_dominates
from optuna.study import Study
from optuna.study._multi_objective import _dominates
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


# Define key names of `Trial.system_attrs`.
_GENERATION_KEY = "nsga3:generation"
_POPULATION_CACHE_KEY_PREFIX = "nsga3:population"


class NSGAIIISampler(BaseSampler):
    """Multi-objective sampler using the NSGA-III algorithm.

    NSGA-III stands for "Nondominated Sorting Genetic Algorithm III",
    which is a modified version of NSGA-II for many objective optimization problem.

    For further information about NSGA-III, please refer to the following papers:

    `An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based
    Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints
    <https://ieeexplore.ieee.org/document/6600851>`_

    `An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based
    Nondominated Sorting Approach, Part II: Handling Constraints and Extending to an Adaptive
    Approach <https://ieeexplore.ieee.org/document/6595567>`_

    Args:
        reference_points:
            A 2 dimension ``numpy.ndarray`` with objective dimension columns. Represents
            a list of reference points which is used to determine who to survive.
            After non-dominated sort, who out of borderline front are going to survived is
            determined according to how sparse the closest reference point of each individual is.
            In the default setting the algorithm uses `uniformly` spread points to diversify the
            result. It is also possible to reflect your `preferences` by giving an arbitrary set of
            `target` points since the algorithm prioritizes individuals around reference points.

            .. note::
                Other parameters are the same as :class:`~optuna.samplers.nsgaii.NSGAIISampler`.

    """

    def __init__(
        self,
        reference_points: np.ndarray,
        population_size: int = 50,
        mutation_prob: Optional[float] = None,
        crossover: Optional[BaseCrossover] = None,
        crossover_prob: float = 0.9,
        swapping_prob: float = 0.5,
        seed: Optional[int] = None,
        constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
    ) -> None:
        # TODO(ohta): Reconsider the default value of each parameter.

        if not isinstance(population_size, int):
            raise TypeError("`population_size` must be an integer value.")

        if population_size < 2:
            raise ValueError("`population_size` must be greater than or equal to 2.")

        if not (mutation_prob is None or 0.0 <= mutation_prob <= 1.0):
            raise ValueError(
                "`mutation_prob` must be None or a float value within the range [0.0, 1.0]."
            )

        if not (0.0 <= crossover_prob <= 1.0):
            raise ValueError("`crossover_prob` must be a float value within the range [0.0, 1.0].")

        if not (0.0 <= swapping_prob <= 1.0):
            raise ValueError("`swapping_prob` must be a float value within the range [0.0, 1.0].")

        if constraints_func is not None:
            warnings.warn(
                "The constraints_func option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )

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

        self.reference_points = reference_points
        self._population_size = population_size
        self._mutation_prob = mutation_prob
        self._crossover = crossover
        self._crossover_prob = crossover_prob
        self._swapping_prob = swapping_prob
        self._random_sampler = RandomSampler(seed=seed)
        self._rng = np.random.RandomState(seed)
        self._constraints_func = constraints_func
        self._search_space = IntersectionSearchSpace()

    def reseed_rng(self) -> None:
        self._random_sampler.reseed_rng()
        self._rng.seed()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:
        search_space: Dict[str, BaseDistribution] = {}
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
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        parent_generation, parent_population = self._collect_parent_population(study)
        trial_id = trial._trial_id

        generation = parent_generation + 1
        study._storage.set_trial_system_attr(trial_id, _GENERATION_KEY, generation)

        dominates_func = _dominates if self._constraints_func is None else _constrained_dominates

        if parent_generation >= 0:
            # We choose a child based on the specified crossover method.
            if self._rng.rand() < self._crossover_prob:
                child_params = perform_crossover(
                    self._crossover,
                    study,
                    parent_population,
                    search_space,
                    self._rng,
                    self._swapping_prob,
                    dominates_func,
                )
            else:
                parent_population_size = len(parent_population)
                parent_params = parent_population[self._rng.choice(parent_population_size)].params
                child_params = {name: parent_params[name] for name in search_space.keys()}

            n_params = len(child_params)
            if self._mutation_prob is None:
                mutation_prob = 1.0 / max(1.0, n_params)
            else:
                mutation_prob = self._mutation_prob

            params = {}
            for param_name in child_params.keys():
                if self._rng.rand() >= mutation_prob:
                    params[param_name] = child_params[param_name]
            return params

        return {}

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

    def _collect_parent_population(self, study: Study) -> Tuple[int, List[FrozenTrial]]:
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
        parent_population: List[FrozenTrial] = []
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
        self, study: Study, population: List[FrozenTrial]
    ) -> List[FrozenTrial]:
        elite_population: List[FrozenTrial] = []
        population_per_rank = self._fast_non_dominated_sort(population, study.directions)
        for population in population_per_rank:
            if len(elite_population) + len(population) < self._population_size:
                elite_population.extend(population)
            else:
                objective_matrix = _normalize(elite_population, population)

                elite_population_num = len(elite_population)
                (
                    nearest_points_count_to_reference_point,
                    reference_point_to_population,
                ) = _associate(objective_matrix, self.reference_points, elite_population_num)

                target_population_size = self._population_size - elite_population_num
                additional_elite_population = _niching(
                    target_population_size,
                    population,
                    nearest_points_count_to_reference_point,
                    reference_point_to_population,
                    self._rng,
                )
                elite_population.extend(additional_elite_population)
                break
        return elite_population

    def _fast_non_dominated_sort(
        self,
        population: List[FrozenTrial],
        directions: List[optuna.study.StudyDirection],
    ) -> List[List[FrozenTrial]]:
        if self._constraints_func is not None:
            for _trial in population:
                _constraints = _trial.system_attrs.get(_CONSTRAINTS_KEY)
                if _constraints is None:
                    continue
                if np.any(np.isnan(np.array(_constraints))):
                    raise ValueError("NaN is not acceptable as constraint value.")

        dominated_count: DefaultDict[int, int] = defaultdict(int)
        dominates_list = defaultdict(list)

        dominates = _dominates if self._constraints_func is None else _constrained_dominates

        for p, q in itertools.combinations(population, 2):
            if dominates(p, q, directions):
                dominates_list[p.number].append(q.number)
                dominated_count[q.number] += 1
            elif dominates(q, p, directions):
                dominates_list[q.number].append(p.number)
                dominated_count[p.number] += 1

        population_per_rank = []
        while population:
            non_dominated_population = []
            i = 0
            while i < len(population):
                if dominated_count[population[i].number] == 0:
                    individual = population[i]
                    if i == len(population) - 1:
                        population.pop()
                    else:
                        population[i] = population.pop()
                    non_dominated_population.append(individual)
                else:
                    i += 1

            for x in non_dominated_population:
                for y in dominates_list[x.number]:
                    dominated_count[y] -= 1

            assert non_dominated_population
            population_per_rank.append(non_dominated_population)

        return population_per_rank

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        assert state in [TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED]
        if self._constraints_func is not None:
            _process_constraints_after_trial(self._constraints_func, study, trial, state)
        self._random_sampler.after_trial(study, trial, state, values)


def multi_choose(n: int, k: int) -> int:
    return math.factorial(n + k - 1) // math.factorial(k) // math.factorial(n - 1)


def generate_default_reference_point(
    objective_dimension: int, dividing_parameter: int = 3
) -> np.ndarray:
    """Generates default reference points which are `uniformly` spread on a hyperplane."""
    reference_points = np.zeros(
        (
            multi_choose(objective_dimension, dividing_parameter),
            objective_dimension,
        )
    )
    for i, comb in enumerate(
        itertools.combinations_with_replacement(range(objective_dimension), dividing_parameter)
    ):
        for j in comb:
            reference_points[i, j] += 1.0
    return reference_points


def _normalize(
    elite_population: List[FrozenTrial],
    population: List[FrozenTrial],
    weights: Optional[np.ndarray] = None,
    method: str = "original",
) -> np.ndarray:
    """Normalizes objective values of population

    An ideal point z* consists of minimums in each axis. Each objective value of population is
    then subtracted by the ideal point.
    An extreme point of each axis is (originally) defined as a minimum solution of achievement
    scalarizing function from the population. After that, intercepts are calculate as intercepts
    of hyperplane which has all the extreme points on it and used to rescale objective values.
    """
    # TODO(Shinichi) Propagate argument "weights" and "method" to the constructor

    # Collect objective values
    objective_dimension = len(population[0].values)
    objective_matrix = np.zeros((len(elite_population + population), objective_dimension))
    for i, trial in enumerate(elite_population + population):
        objective_matrix[i] = np.array(trial.values, dtype=float)

    # Subtract ideal point
    objective_matrix -= np.min(objective_matrix, axis=0)
    objective_matrix[objective_matrix < 10e-3] = 0.0

    # TODO(Shinichi) Find out exact definition of "extreme point."
    # weights are m vectors in m dimension where m is the dimension of the objective.
    # weights can be anything as long as the i-th vector is close to each i-th objective axis.

    # Initialize weight
    if weights is None:
        weights = np.eye(objective_dimension)
        if method == "original":
            weights[weights == 0] = 1e6

    # Calculate extreme points to normalize objective values
    if method == "custom":
        # Note that the original paper says that chose ASF "minima" but no exact definition is
        # provided in the paper.
        asf_value = objective_matrix @ weights
        extreme_points = objective_matrix[np.argmax(asf_value, axis=0)]
    elif method == "original":
        # TODO(Shinichi) Reimplement to reduce time complexity
        # Implementation from pymoo, which is the official implementation of the paper
        asf_value = np.max(objective_matrix * weights[:, np.newaxis, :], axis=2)
        extreme_points = objective_matrix[np.argmin(asf_value, axis=1), :]
    else:
        raise ValueError("method should be either custom or original")

    # Normalize objective_matrix with extreme points.
    # Note that extreme_points can be degenerate, but no proper operation is remarked in the
    # paper. Therefore, the maximum of elite population in each axis is used in the case.
    if np.linalg.matrix_rank(extreme_points) < len(extreme_points):
        intercepts_inv = 1 / np.max(objective_matrix[:, :], axis=0)
    else:
        intercepts_inv = np.linalg.solve(extreme_points, np.ones(objective_dimension))
    objective_matrix *= intercepts_inv

    return objective_matrix


def _associate(
    objective_matrix: np.ndarray, reference_points: np.ndarray, elite_population_num: int
) -> Tuple[Dict[int, List[int]], Dict[int, List[Tuple[float, int]]]]:
    """Associates each objective value to the closest reference point"""
    # TODO(Shinichi) Implement faster assignment for the default reference points because it does
    # not seem necessary to calculate distance from all reference points.

    # TODO(Shinichi) Normalize reference_points in constructor to remove reference_point_norms.
    # In addition, the minimum distance from each reference point can be replace with maximum inner
    # product between the given point and each normalized reference points.
    reference_point_norm_squared = np.sum(reference_points**2, axis=1)
    coefficient = objective_matrix @ reference_points.T / reference_point_norm_squared
    distance_from_reference_lines = np.linalg.norm(
        objective_matrix[:, np.newaxis, :] - coefficient[..., np.newaxis] * reference_points,
        axis=2,
    )
    distance_reference_points = np.min(distance_from_reference_lines, axis=1)
    closest_reference_points = np.argmin(distance_from_reference_lines, axis=1)

    # count_reference_points keeps how many elite neighbors each reference point has
    count_reference_points: Dict[int, int] = DefaultDict(int)
    for i, reference_point_id in enumerate(closest_reference_points[:elite_population_num]):
        count_reference_points[reference_point_id] += 1

    # reference_point_to_population keeps pairs of a neighbor and the distance of each reference
    # point from borderline front population
    reference_point_to_population = DefaultDict(list)
    for i, reference_point_id in enumerate(closest_reference_points[elite_population_num:]):
        reference_point_to_population[reference_point_id].append(
            (distance_reference_points[i + elite_population_num], i)
        )

    # nearest_points_count_to_reference_point classifies reference points which have at least one
    # closest borderline population member by the number of elite neighbors they have. Each key
    # correspond to the count of elite neighbors and values are reference point ids.
    nearest_points_count_to_reference_point = DefaultDict(list)
    for reference_point_id in reference_point_to_population:
        count = count_reference_points[reference_point_id]
        nearest_points_count_to_reference_point[count].append(reference_point_id)

    return nearest_points_count_to_reference_point, reference_point_to_population


def _niching(
    target_population_size: int,
    population: List[FrozenTrial],
    nearest_points_count_to_reference_point: Dict[int, List[int]],
    reference_point_to_population: Dict[int, List[Tuple[float, int]]],
    rng: np.random.RandomState,
) -> List[FrozenTrial]:
    """Determine who survives form the borderline front

    Who survive form the borderline front is determined according to the sparsity of each closest
    reference point. The algorithm picks a reference point from those who have the least neighbors
    in elite population and adds one of borderline front member who has the same closest reference
    point.
    """

    count = 0
    additional_elite_population: List[FrozenTrial] = []
    while len(additional_elite_population) < target_population_size:
        if not nearest_points_count_to_reference_point[count]:
            count += 1
            continue

        individual_idx = rng.choice(len(nearest_points_count_to_reference_point[count]))
        (
            nearest_points_count_to_reference_point[count][individual_idx],
            nearest_points_count_to_reference_point[count][-1],
        ) = (
            nearest_points_count_to_reference_point[count][-1],
            nearest_points_count_to_reference_point[count][individual_idx],
        )

        reference_point_id = nearest_points_count_to_reference_point[count].pop()
        if count:
            reference_point_idx = rng.choice(
                len(reference_point_to_population[reference_point_id])
            )
            (
                reference_point_to_population[reference_point_id][reference_point_idx],
                reference_point_to_population[reference_point_id][-1],
            ) = (
                reference_point_to_population[reference_point_id][-1],
                reference_point_to_population[reference_point_id][reference_point_idx],
            )
        else:
            # TODO(Shinichi) avoid sort
            reference_point_to_population[reference_point_id].sort(reverse=True)

        _, selected_individual_id = reference_point_to_population[reference_point_id].pop()
        additional_elite_population.append(population[selected_individual_id])
        if reference_point_to_population[reference_point_id]:
            nearest_points_count_to_reference_point[count + 1].append(reference_point_id)

    return additional_elite_population
