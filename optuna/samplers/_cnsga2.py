from collections import defaultdict
import copy
import itertools
from typing import Callable
from typing import DefaultDict
from typing import List
from typing import Optional
from typing import Sequence
import warnings

import optuna
from optuna._multi_objective import _dominates
from optuna.samplers import NSGAIISampler
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState

_CONSTRAINTS_KEY = "cnsga2:constraints"


class CNSGAIISampler(NSGAIISampler):
    def __init__(
        self,
        *,
        population_size: int = 50,
        mutation_prob: Optional[float] = None,
        crossover_prob: float = 0.9,
        swapping_prob: float = 0.5,
        seed: Optional[int] = None,
        constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
    ) -> None:
        super().__init__(
            population_size=population_size,
            mutation_prob=mutation_prob,
            crossover_prob=crossover_prob,
            swapping_prob=swapping_prob,
            seed=seed,
        )
        self._constraints_func = constraints_func

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Optional[Sequence[float]],
    ) -> None:
        if self._constraints_func is not None:
            constraints = None
            _trial = copy.copy(trial)
            _trial.state = state
            _trial.values = values
            try:
                con = self._constraints_func(_trial)
                if not isinstance(con, (tuple, list)):
                    warnings.warn(
                        f"Constraints should be a sequence of floats but got {constraints}."
                    )
                constraints = tuple(con)
            except Exception:
                raise
            finally:
                assert constraints is None or isinstance(constraints, tuple)

                study._storage.set_trial_system_attr(
                    trial._trial_id,
                    _CONSTRAINTS_KEY,
                    constraints,
                )

    def _select_elite_population(
        self, study: Study, population: List[FrozenTrial]
    ) -> List[FrozenTrial]:
        elite_population: List[FrozenTrial] = []
        population_per_rank = _fast_non_dominated_sort(population, study.directions)
        for population in population_per_rank:
            if len(elite_population) + len(population) < self._population_size:
                elite_population.extend(population)
            else:
                n = self._population_size - len(elite_population)
                optuna.samplers._nsga2._crowding_distance_sort(population)
                elite_population.extend(population[:n])
                break

        return elite_population

    def _select_parent(self, study: Study, population: List[FrozenTrial]) -> FrozenTrial:
        # TODO(ohta): Consider to allow users to specify the number of parent candidates.
        candidate0 = self._rng.choice(population)
        candidate1 = self._rng.choice(population)

        # TODO(ohta): Consider crowding distance.
        if _sigma_dominates(candidate0, candidate1, study.directions):
            return candidate0
        else:
            return candidate1


def _sigma_dominates(
    trial0: FrozenTrial, trial1: FrozenTrial, directions: Sequence[StudyDirection]
) -> bool:
    constraints0 = trial0.system_attrs[_CONSTRAINTS_KEY]
    constraints1 = trial1.system_attrs[_CONSTRAINTS_KEY]

    assert isinstance(constraints0, (list, tuple))
    assert isinstance(constraints1, (list, tuple))

    if len(constraints1) != len(constraints1):
        raise ValueError("Trials with different numbers of constraints cannot be compared.")

    if trial0.state != TrialState.COMPLETE:
        return False

    if trial1.state != TrialState.COMPLETE:
        return True

    if all(v <= 0 for v in constraints0) and all(v <= 0 for v in constraints1):
        # Both trials satisfy the constraints.
        return _dominates(trial0, trial1, directions)

    if all(v <= 0 for v in constraints0):
        # trial0 satisfies the constraints, but trial1 violates them.
        return True

    if all(v <= 0 for v in constraints1):
        # trial1 satisfies the constraints, but trial0 violates them.
        return False

    # Both trials violate the constraints.
    violation0 = sum(v for v in constraints0 if v > 0)
    violation1 = sum(v for v in constraints1 if v > 0)
    return violation0 < violation1


def _fast_non_dominated_sort(
    population: List[FrozenTrial],
    directions: List[StudyDirection],
) -> List[List[FrozenTrial]]:
    dominated_count: DefaultDict[int, int] = defaultdict(int)
    dominates_list = defaultdict(list)

    for p, q in itertools.combinations(population, 2):
        if _sigma_dominates(p, q, directions):
            dominates_list[p.number].append(q.number)
            dominated_count[q.number] += 1
        elif _sigma_dominates(q, p, directions):
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
