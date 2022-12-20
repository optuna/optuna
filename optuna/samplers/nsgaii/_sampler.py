from collections import defaultdict
import hashlib
import itertools
from typing import Any
from typing import Callable
from typing import cast
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
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.study._multi_objective import _dominates
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


# Define key names of `Trial.system_attrs`.
_GENERATION_KEY = "nsga2:generation"
_POPULATION_CACHE_KEY_PREFIX = "nsga2:population"


class NSGAIISampler(BaseSampler):
    """Multi-objective sampler using the NSGA-II algorithm.

    NSGA-II stands for "Nondominated Sorting Genetic Algorithm II",
    which is a well known, fast and elitist multi-objective genetic algorithm.

    For further information about NSGA-II, please refer to the following paper:

    - `A fast and elitist multiobjective genetic algorithm: NSGA-II
      <https://ieeexplore.ieee.org/document/996017>`_

    Args:
        population_size:
            Number of individuals (trials) in a generation.
            ``population_size`` must be greater than or equal to ``crossover.n_parents``.
            For :class:`~optuna.samplers.nsgaii.UNDXCrossover` and
            :class:`~optuna.samplers.nsgaii.SPXCrossover`, ``n_parents=3``, and for the other
            algorithms, ``n_parents=2``.

        mutation_prob:
            Probability of mutating each parameter when creating a new individual.
            If :obj:`None` is specified, the value ``1.0 / len(parent_trial.params)`` is used
            where ``parent_trial`` is the parent trial of the target individual.

        crossover:
            Crossover to be applied when creating child individuals.
            The available crossovers are listed here:
            https://optuna.readthedocs.io/en/stable/reference/samplers/nsgaii.html.

            :class:`~optuna.samplers.nsgaii.UniformCrossover` is always applied to parameters
            sampled from :class:`~optuna.distributions.CategoricalDistribution`, and by
            default for parameters sampled from other distributions unless this argument
            is specified.

            For more information on each of the crossover method, please refer to
            specific crossover documentation.

        crossover_prob:
            Probability that a crossover (parameters swapping between parents) will occur
            when creating a new individual.

        swapping_prob:
            Probability of swapping each parameter of the parents during crossover.

        seed:
            Seed for random number generator.

        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraints is violated. A value equal to or smaller than 0 is considered feasible.
            If ``constraints_func`` returns more than one value for a trial, that trial is
            considered feasible if and only if all values are equal to 0 or smaller.

            The ``constraints_func`` will be evaluated after each successful trial.
            The function won't be called when trials fail or they are pruned, but this behavior is
            subject to change in the future releases.

            The constraints are handled by the constrained domination. A trial x is said to
            constrained-dominate a trial y, if any of the following conditions is true:

            1. Trial x is feasible and trial y is not.
            2. Trial x and y are both infeasible, but trial x has a smaller overall violation.
            3. Trial x and y are feasible and trial x dominates trial y.

            .. note::
                Added in v2.5.0 as an experimental feature. The interface may change in newer
                versions without prior notice. See
                https://github.com/optuna/optuna/releases/tag/v2.5.0.

    """

    def __init__(
        self,
        *,
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
                n = self._population_size - len(elite_population)
                _crowding_distance_sort(population)
                elite_population.extend(population[:n])
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


def _calc_crowding_distance(population: List[FrozenTrial]) -> DefaultDict[int, float]:
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

    manhattan_distances: DefaultDict[int, float] = defaultdict(float)
    if len(population) == 0:
        return manhattan_distances

    for i in range(len(population[0].values)):
        population.sort(key=lambda x: cast(float, x.values[i]))

        # If population have the same values[i], ignore that value.
        if population[0].values[i] == population[-1].values[i]:
            continue

        vs = (
            [-float("inf")]
            + [cast(List[float], population[j].values)[i] for j in range(len(population))]
            + [float("inf")]
        )

        # Smallest finite value.
        v_min = next(x for x in vs if x != -float("inf"))

        # Largest finite value.
        v_max = next(x for x in reversed(vs) if x != float("inf"))

        width = v_max - v_min
        if width <= 0:
            # width == 0 or width == -inf
            width = 1.0

        for j in range(len(population)):
            # inf - inf and (-inf) - (-inf) is considered to be zero.
            gap = 0.0 if vs[j] == vs[j + 2] else vs[j + 2] - vs[j]
            manhattan_distances[population[j].number] += gap / width
    return manhattan_distances


def _crowding_distance_sort(population: List[FrozenTrial]) -> None:
    manhattan_distances = _calc_crowding_distance(population)
    population.sort(key=lambda x: manhattan_distances[x.number])
    population.reverse()


def _constrained_dominates(
    trial0: FrozenTrial, trial1: FrozenTrial, directions: Sequence[StudyDirection]
) -> bool:
    """Checks constrained-domination.

    A trial x is said to constrained-dominate a trial y, if any of the following conditions is
    true:
    1) Trial x is feasible and trial y is not.
    2) Trial x and y are both infeasible, but solution x has a smaller overall constraint
    violation.
    3) Trial x and y are feasible and trial x dominates trial y.
    """

    constraints0 = trial0.system_attrs.get(_CONSTRAINTS_KEY)
    constraints1 = trial1.system_attrs.get(_CONSTRAINTS_KEY)

    if constraints0 is None:
        warnings.warn(
            f"Trial {trial0.number} does not have constraint values."
            " It will be dominated by the other trials."
        )

    if constraints1 is None:
        warnings.warn(
            f"Trial {trial1.number} does not have constraint values."
            " It will be dominated by the other trials."
        )

    if constraints0 is None and constraints1 is None:
        # Neither Trial x nor y has constraints values
        return _dominates(trial0, trial1, directions)

    if constraints0 is not None and constraints1 is None:
        # Trial x has constraint values, but y doesn't.
        return True

    if constraints0 is None and constraints1 is not None:
        # If Trial y has constraint values, but x doesn't.
        return False

    assert isinstance(constraints0, (list, tuple))
    assert isinstance(constraints1, (list, tuple))

    if len(constraints0) != len(constraints1):
        raise ValueError("Trials with different numbers of constraints cannot be compared.")

    if trial0.state != TrialState.COMPLETE:
        return False

    if trial1.state != TrialState.COMPLETE:
        return True

    satisfy_constraints0 = all(v <= 0 for v in constraints0)
    satisfy_constraints1 = all(v <= 0 for v in constraints1)

    if satisfy_constraints0 and satisfy_constraints1:
        # Both trials satisfy the constraints.
        return _dominates(trial0, trial1, directions)

    if satisfy_constraints0:
        # trial0 satisfies the constraints, but trial1 violates them.
        return True

    if satisfy_constraints1:
        # trial1 satisfies the constraints, but trial0 violates them.
        return False

    # Both trials violate the constraints.
    violation0 = sum(v for v in constraints0 if v > 0)
    violation1 = sum(v for v in constraints1 if v > 0)
    return violation0 < violation1
