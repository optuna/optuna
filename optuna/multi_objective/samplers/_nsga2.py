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


GENERATION_KEY = "multi_objective:nsga2:generation"
PARENTS_KEY = "multi_objective:nsga2:parents"


@experimental("1.4.0")
class NSGAIIMultiObjectiveSampler(BaseMultiObjectiveSampler):
    def __init__(
        self,
        seed: Optional[int] = None,
        population_size: int = 20,
        crossover_prob: float = 0.5,
        mutation_prob: float = 0.3,
    ) -> None:
        self._population_size = population_size
        self._crossover_prob = crossover_prob
        self._mutation_prob = mutation_prob
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
        parent_generation, population = self._collect_parent_population(study)
        trial_id = trial._trial_id

        if len(population) == 0:
            generation = 0
            study._storage.set_trial_system_attr(trial_id, GENERATION_KEY, generation)
        else:
            p0 = self._select_parent(study, population)
            p1 = self._select_parent(study, population)

            generation = parent_generation + 1
            study._storage.set_trial_system_attr(trial_id, GENERATION_KEY, generation)
            study._storage.set_trial_system_attr(trial_id, PARENTS_KEY, [p0.number, p1.number])

        return {}

    def sample_independent(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        trial: "multi_objective.trial.FrozenMultiObjectiveTrial",
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if PARENTS_KEY not in trial.system_attrs:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        trials = study.get_trials(deepcopy=False)
        p0, p1 = trial.system_attrs[PARENTS_KEY]

        param = trials[p0].params.get(param_name, None)
        if param is None or self._rng.rand() < self._crossover_prob:
            param = trials[p1].params.get(param_name, None)

        if param is None or self._rng.rand() < self._mutation_prob:
            return self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        return param

    def _collect_parent_population(
        self, study: "multi_objective.study.MultiObjectiveStudy"
    ) -> Tuple[int, List["multi_objective.trial.FrozenMultiObjectiveTrial"]]:
        trials = study.get_trials(deepcopy=False)
        parent_population = []  # type: List[multi_objective.trial.FrozenMultiObjectiveTrial]
        parent_generation = None
        for generation in itertools.count():
            population = [
                t
                for t in trials
                if (
                    t.system_attrs.get(GENERATION_KEY, 0) == generation
                    and t.state == optuna.trial.TrialState.COMPLETE
                )
            ]
            if len(population) < self._population_size:
                break

            population.extend(parent_population)
            parent_population = []
            parent_generation = generation

            population_per_rank = self._fast_non_dominated_sort(study, population)
            for population in population_per_rank:
                if len(parent_population) + len(population) < self._population_size:
                    parent_population.extend(population)
                else:
                    n = self._population_size - len(parent_population)
                    self._crowding_distance_sort(population)
                    parent_population.extend(population[:n])
                    break

        assert parent_generation is not None
        return parent_generation, parent_population

    def _fast_non_dominated_sort(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        population: List["multi_objective.trial.FrozenMultiObjectiveTrial"],
    ) -> List[List["multi_objective.trial.FrozenMultiObjectiveTrial"]]:
        dominated_count = defaultdict(int)  # type: DefaultDict[int, int]
        dominates_list = defaultdict(list)

        for p, q in itertools.combinations(population, 2):
            if p._dominates(q, study.directions):
                dominates_list[p.number].append(q.number)
                dominated_count[q.number] += 1
            elif q._dominates(p, study.directions):
                dominates_list[q.number].append(p.number)
                dominated_count[p.number] += 1

        population_per_rank = []
        while len(population) > 0:
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

            assert non_dominated_population != []
            population_per_rank.append(non_dominated_population)

        return population_per_rank

    def _crowding_distance_sort(
        self, population: List["multi_objective.trial.FrozenMultiObjectiveTrial"]
    ) -> None:
        distances = defaultdict(float)
        for i in range(len(population[0].values)):
            population.sort(key=lambda x: x.values[i])

            distances[population[0].number] = float("inf")
            distances[population[-1].number] = float("inf")

            v_max = population[-1].values[i]
            v_min = population[0].values[i]
            assert v_max is not None
            assert v_min is not None

            width = v_max - v_min

            for j in range(1, len(population) - 1):
                v_high = population[j + 1].values[i]
                v_low = population[j - 1].values[i]
                assert v_high is not None
                assert v_low is not None

                distances[population[j].number] += (v_high - v_low) / width

        population.sort(key=lambda x: distances[x.number])
        population.reverse()

    def _select_parent(
        self,
        study: "multi_objective.study.MultiObjectiveStudy",
        population: List["multi_objective.trial.FrozenMultiObjectiveTrial"],
    ) -> "multi_objective.trial.FrozenMultiObjectiveTrial":
        candidate0 = self._rng.choice(population)
        candidate1 = self._rng.choice(population)
        # TODO: Consider crowding distance.
        if candidate0._dominates(candidate1, study.directions):
            return candidate0
        else:
            return candidate1
