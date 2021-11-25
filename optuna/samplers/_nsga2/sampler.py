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
from optuna._experimental import ExperimentalWarning
from optuna.distributions import BaseDistribution
from optuna.samplers._base import BaseSampler
from optuna.samplers._nsga2.crossover import crossover
from optuna.samplers._nsga2.crossover import get_n_parents
from optuna.samplers._random import RandomSampler
from optuna.samplers._search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.study import StudyDirection
from optuna.study._multi_objective import _dominates
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


# Define key names of `Trial.system_attrs`.
_CONSTRAINTS_KEY = "nsga2:constraints"
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

        mutation_prob:
            Probability of mutating each parameter when creating a new individual.
            If :obj:`None` is specified, the value ``1.0 / len(parent_trial.params)`` is used
            where ``parent_trial`` is the parent trial of the target individual.

        crossover:
            Crossover to be applied when creating child individuals.
            The available crossovers are
            `uniform` (default), `blxalpha`, `sbx`, `vsbx`, `undx`, and `spx`.

            For :class:`~optuna.distributions.CategoricalDistribution`,
            uniform crossover will be applied,
            and for other distributions, the specified crossover will be applied.

            For more information on each of the crossover method, please refer to the following.

            - uniform: Select each parameter with equal probability
              from the two parent individuals.

                - `Gilbert Syswerda. 1989. Uniform Crossover in Genetic Algorithms.
                  In Proceedings of the 3rd International Conference on Genetic Algorithms.
                  Morgan Kaufmann Publishers Inc., San Francisco, CA, USA, 2–9.
                  <https://www.researchgate.net/publication/201976488_Uniform_Crossover_in_Genetic_Algorithms>`_

            - blxalpha: It uniformly samples child individuals from the hyper-rectangles created
              by the two parent individuals.

                - `Eshelman, L. and J. D. Schaffer.
                  Real-Coded Genetic Algorithms and Interval-Schemata. FOGA (1992).
                  <https://www.sciencedirect.com/science/article/abs/pii/B9780080948324500180>`_

            - sbx: Generate a child from two parent individuals
              according to the polynomial probability distribution.

                - `Deb, K. and R. Agrawal.
                  “Simulated Binary Crossover for Continuous Search Space.”
                  Complex Syst. 9 (1995): n. pag.
                  <https://www.complex-systems.com/abstracts/v09_i02_a02/>`_

            - vsbx: In SBX, the probability of occurrence of child individuals
              becomes zero in some parameter regions.
              vSBX generates child individuals without excluding any region of the parameter space,
              while maintaining the excellent properties of SBX.

                - `Pedro J. Ballester, Jonathan N. Carter.
                  Real-Parameter Genetic Algorithms for Finding Multiple Optimal Solutions
                  in Multi-modal Optimization. GECCO 2003: 706-717
                  <https://link.springer.com/chapter/10.1007/3-540-45105-6_86>`_

            - undx: Generate child individuals from the three parents
              using a multivariate normal distribution.

                - `H. Kita, I. Ono and S. Kobayashi,
                  Multi-parental extension of the unimodal normal distribution crossover
                  for real-coded genetic algorithms,
                  Proceedings of the 1999 Congress on Evolutionary Computation-CEC99
                  (Cat. No. 99TH8406), 1999, pp. 1581-1588 Vol. 2
                  <https://ieeexplore.ieee.org/document/782672>`_

            - spx: It uniformly samples child individuals from within a single simplex
              that is similar to the simplex produced by the parent individual.

                - `Shigeyoshi Tsutsui and Shigeyoshi Tsutsui and David E. Goldberg and
                  David E. Goldberg and Kumara Sastry and Kumara Sastry
                  Progress Toward Linkage Learning in Real-Coded GAs with Simplex Crossover.
                  IlliGAL Report. 2000.
                  <https://www.researchgate.net/publication/2388486_Progress_Toward_Linkage_Learning_in_Real-Coded_GAs_with_Simplex_Crossover>`_

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

    Raises:
        ValueError:
            If ``crossover`` is not in `[uniform, blxalpha, sbx, vsbx, undx, spx]`.
            Or, if ``population_size <= n_parents``.
            The `n_parents` is determined by each crossover.
            For `undx` and `spx`, ``n_parents=3``, and for the other algorithms, ``n_parents=2``.
    """

    def __init__(
        self,
        *,
        population_size: int = 50,
        mutation_prob: Optional[float] = None,
        crossover: str = "uniform",
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
        if crossover not in ["uniform", "blxalpha", "sbx", "vsbx", "undx", "spx"]:
            raise ValueError(
                f"'{crossover}' is not a valid crossover name."
                " The available crossovers are"
                " `uniform` (default), `blxalpha`, `sbx`, `vsbx`, `undx`, and `spx`."
            )
        if crossover != "uniform":
            warnings.warn(
                "``crossover`` option is an experimental feature."
                " The interface can change in the future.",
                ExperimentalWarning,
            )
        n_parents = get_n_parents(crossover)
        if population_size < n_parents:
            raise ValueError(
                f"Using {crossover},"
                f" the population size should be greater than or equal to {n_parents}."
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
        self._rng = np.random.RandomState()

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
                child_params = crossover(
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
        if state == TrialState.COMPLETE and self._constraints_func is not None:
            constraints = None
            try:
                con = self._constraints_func(trial)
                if not isinstance(con, (tuple, list)):
                    warnings.warn(
                        f"Constraints should be a sequence of floats but got {type(con).__name__}."
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
        self._random_sampler.after_trial(study, trial, state, values)


def _crowding_distance_sort(population: List[FrozenTrial]) -> None:
    manhattan_distances = defaultdict(float)
    for i in range(len(population[0].values)):
        population.sort(key=lambda x: cast(float, x.values[i]))

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
