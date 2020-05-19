from collections import defaultdict
import itertools
from typing import Any
from typing import DefaultDict
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

import optuna
from optuna._experimental import experimental
from optuna.distributions import BaseDistribution
from optuna import multi_objective
from optuna.multi_objective.samplers import BaseMultiObjectiveSampler


# Define key names of `Trial.system_attrs`.
_GENERATION_KEY = "multi_objective:nsga2:generation"
_PARENTS_KEY = "multi_objective:nsga2:parents"


@experimental("1.5.0")
class NSGAIIMultiObjectiveSampler(BaseMultiObjectiveSampler):
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
        self._random_sampler = multi_objective.samplers.RandomMultiObjectiveSampler(seed=seed)
        self._rng = np.random.RandomState(seed)

    def infer_relative_search_space(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
    ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
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
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
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

    def _collect_parent_population(
        self, study: "multi_objective.study.MultiObjectiveStudy"
    ) -> Tuple[int, List["multi_objective.trial.FrozenMultiObjectiveTrial"]]:
        # TODO(ohta): Optimize this method.

        generation_to_population = defaultdict(list)
        for trial in study.get_trials(deepcopy=False):
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            generation = trial.system_attrs.get(_GENERATION_KEY, 0)
            generation_to_population[generation].append(trial)

        parent_population = []  # type: List[multi_objective.trial.FrozenMultiObjectiveTrial]
        parent_generation = -1
        for generation in itertools.count():
            population = generation_to_population[generation]

            # Under multi-worker settings, the population size might become larger than
            # `self._population_size`.
            if len(population) < self._population_size:
                break

            population.extend(parent_population)
            parent_population = []
            parent_generation = generation

            population_per_rank = _fast_non_dominated_sort(population, study.directions)
            for population in population_per_rank:
                if len(parent_population) + len(population) < self._population_size:
                    parent_population.extend(population)
                else:
                    n = self._population_size - len(parent_population)
                    _crowding_distance_sort(population)
                    parent_population.extend(population[:n])
                    break

        return parent_generation, parent_population

    def _select_parent(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        population: List["multi_objective.trial.FrozenMultiObjectiveTrial"],
    ) -> "multi_objective.trial.FrozenMultiObjectiveTrial":
        # TODO(ohta): Consider to allow users to specify the number of parent candidates.
        candidate0 = self._rng.choice(population)
        candidate1 = self._rng.choice(population)

        # TODO(ohta): Consider crowding distance.
        if candidate0._dominates(candidate1, study.directions):
            return candidate0
        else:
            return candidate1


def _fast_non_dominated_sort(
    population: List["multi_objective.trial.FrozenMultiObjectiveTrial"],
    directions: List[optuna.study.StudyDirection],
) -> List[List["multi_objective.trial.FrozenMultiObjectiveTrial"]]:
    dominated_count = defaultdict(int)  # type: DefaultDict[int, int]
    dominates_list = defaultdict(list)

    for p, q in itertools.combinations(population, 2):
        if p._dominates(q, directions):
            dominates_list[p.number].append(q.number)
            dominated_count[q.number] += 1
        elif q._dominates(p, directions):
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
    population: List["multi_objective.trial.FrozenMultiObjectiveTrial"],
) -> None:
    manhattan_distances = defaultdict(float)
    for i in range(len(population[0].values)):
        population.sort(key=lambda x: x.values[i])

        v_min = population[0].values[i]
        v_max = population[-1].values[i]
        assert v_min is not None
        assert v_max is not None

        width = v_max - v_min
        if width == 0:
            continue

        manhattan_distances[population[0].number] = float("inf")
        manhattan_distances[population[-1].number] = float("inf")

        for j in range(1, len(population) - 1):
            v_high = population[j + 1].values[i]
            v_low = population[j - 1].values[i]
            assert v_high is not None
            assert v_low is not None

            manhattan_distances[population[j].number] += (v_high - v_low) / width

    population.sort(key=lambda x: manhattan_distances[x.number])
    population.reverse()
