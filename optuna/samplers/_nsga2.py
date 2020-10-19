from collections import defaultdict
import hashlib
import itertools
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

import optuna
from optuna import multi_objective
from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler
from optuna.multi_objective._selection import _dominates


# Define key names of `Trial.system_attrs`.
_GENERATION_KEY = "multi_objective:nsga2:generation"
_PARENTS_KEY = "multi_objective:nsga2:parents"
_POPULATION_CACHE_KEY_PREFIX = "multi_objective:nsga2:population"


@experimental("2.2.0")
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

        mutation_prob:
            Probability of mutating each parameter when creating a new individual.
            If :obj:`None` is specified, the value ``1.0 / len(parent_trial.params)`` is used
            where ``parent_trial`` is the parent trial of the target individual.

        crossover_prob:
            Probability that a crossover (parameters swapping between parents) will occur
            when creating a new individual.

        swapping_prob:
            Probability of swapping each parameter of the parents during crossover.

        seed:
            Seed for random number generator.

    """

    def __init__(
        self,
        population_size: int = 50,
        mutation_prob: Optional[float] = None,
        crossover_prob: float = 0.9,
        swapping_prob: float = 0.5,
        seed: Optional[int] = None,
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

        self._population_size = population_size
        self._mutation_prob = mutation_prob
        self._crossover_prob = crossover_prob
        self._swapping_prob = swapping_prob
        self._random_sampler = optuna.samplers.RandomSampler(seed=seed)
        self._rng = np.random.RandomState(seed)

    def infer_relative_search_space(
        self,
        study: "Study",
        trial: "FrozenTrial",
    ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: "Study",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        parent_generation, parent_population = self._collect_parent_population(study)
        trial_id = trial._trial_id

        generation = parent_generation + 1
        study._storage.set_trial_system_attr(trial_id, _GENERATION_KEY, generation)

        if parent_generation >= 0:
            p0 = self._select_parent(study, parent_population)
            if self._rng.rand() < self._crossover_prob:
                p1 = self._select_parent(
                    study, [t for t in parent_population if t._trial_id != p0._trial_id]
                )
            else:
                p1 = p0

            study._storage.set_trial_system_attr(
                trial_id, _PARENTS_KEY, [p0._trial_id, p1._trial_id]
            )

        return {}

    def sample_independent(
        self,
        study: "Study",
        trial: "FrozenTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if _PARENTS_KEY not in trial.system_attrs:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        p0_id, p1_id = trial.system_attrs[_PARENTS_KEY]
        p0 = study._storage.get_trial(p0_id)
        p1 = study._storage.get_trial(p1_id)

        param = p0.params.get(param_name, None)
        parent_params_len = len(p0.params)
        if param is None or self._rng.rand() < self._swapping_prob:
            param = p1.params.get(param_name, None)
            parent_params_len = len(p1.params)

        mutation_prob = self._mutation_prob
        if mutation_prob is None:
            mutation_prob = 1.0 / max(1.0, parent_params_len)

        if param is None or self._rng.rand() < mutation_prob:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        return param

    def _collect_parent_population(self, study: "Study") -> Tuple[int, List["FrozenTrial"]]:
        trials = study._storage.get_all_trials(study._study_id, deepcopy=False)

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

            generation_to_population[generation].append(trial)

        hasher = hashlib.sha256()
        parent_population = []  # type: List[FrozenTrial]
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
            cached_generation, cached_population_numbers = study.system_attrs.get(
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
                    study.set_system_attr(cache_key, (generation, population_numbers))

            parent_generation = generation
            parent_population = population

        return parent_generation, parent_population

    def _select_elite_population(
        self,
        study: "Study",
        population: List["FrozenTrial"],
    ) -> List["FrozenTrial"]:
        elite_population = []  # type: List[multi_objective.trial.FrozenMultiObjectiveTrial]
        population_per_rank = _fast_non_dominated_sort(population, study.direction)
        for population in population_per_rank:
            if len(elite_population) + len(population) < self._population_size:
                elite_population.extend(population)
            else:
                n = self._population_size - len(elite_population)
                _crowding_distance_sort(population)
                elite_population.extend(population[:n])
                break

        return elite_population

    def _select_parent(
        self,
        study: "Study",
        population: List["FrozenTrial"],
    ) -> "FrozenTrial":
        # TODO(ohta): Consider to allow users to specify the number of parent candidates.
        candidate0 = self._rng.choice(population)
        candidate1 = self._rng.choice(population)

        # TODO(ohta): Consider crowding distance.
        if _dominates(candidate0, candidate1, study.direction):
            return candidate0
        else:
            return candidate1


def _fast_non_dominated_sort(
    population: List["FrozenTrial"],
    directions: List[optuna.study.StudyDirection],
) -> List[List["FrozenTrial"]]:
    dominated_count = defaultdict(int)  # type: DefaultDict[int, int]
    dominates_list = defaultdict(list)

    for p, q in itertools.combinations(population, 2):
        if _dominates(p, q, directions):
            dominates_list[p.number].append(q.number)
            dominated_count[q.number] += 1
        elif _dominates(q, p, directions):
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


def _crowding_distance_sort(
    population: List["FrozenTrial"],
) -> None:
    manhattan_distances = defaultdict(float)
    for i in range(len(population[0].value)):
        population.sort(key=lambda x: x.value[i])

        v_min = population[0].value[i]
        v_max = population[-1].value[i]
        assert v_min is not None
        assert v_max is not None

        width = v_max - v_min
        if width == 0:
            continue

        manhattan_distances[population[0].number] = float("inf")
        manhattan_distances[population[-1].number] = float("inf")

        for j in range(1, len(population) - 1):
            v_high = population[j + 1].value[i]
            v_low = population[j - 1].value[i]
            assert v_high is not None
            assert v_low is not None

            manhattan_distances[population[j].number] += (v_high - v_low) / width

    population.sort(key=lambda x: manhattan_distances[x.number])
    population.reverse()
