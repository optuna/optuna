from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from optuna.samplers.nsgaii._constraints_evaluation import _constrained_dominates
from optuna.samplers.nsgaii._crossover import perform_crossover
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.samplers.nsgaii._mutation import perform_mutation
from optuna.samplers.nsgaii._mutations._base import BaseMutation
from optuna.study._constrained_optimization import _is_constrained_optimization
from optuna.study._multi_objective import _dominates


if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from optuna.distributions import BaseDistribution
    from optuna.samplers._lazy_random_state import LazyRandomState
    from optuna.study import Study
    from optuna.trial import FrozenTrial


class NSGAIIChildGenerationStrategy:
    def __init__(
        self,
        *,
        mutation: BaseMutation | None = None,
        mutation_prob: float | None = None,
        crossover: BaseCrossover,
        crossover_prob: float,
        swapping_prob: float,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        rng: LazyRandomState,
    ) -> None:
        if not (mutation_prob is None or 0.0 <= mutation_prob <= 1.0):
            raise ValueError(
                "`mutation_prob` must be None or a float value within the range [0.0, 1.0]."
            )

        if not (0.0 <= crossover_prob <= 1.0):
            raise ValueError("`crossover_prob` must be a float value within the range [0.0, 1.0].")

        if not (0.0 <= swapping_prob <= 1.0):
            raise ValueError("`swapping_prob` must be a float value within the range [0.0, 1.0].")

        if mutation is not None and not isinstance(mutation, BaseMutation):
            raise ValueError(f"'{mutation}' is not a valid mutation.")

        if not isinstance(crossover, BaseCrossover):
            raise ValueError(
                f"'{crossover}' is not a valid crossover."
                " For valid crossovers see"
                " https://optuna.readthedocs.io/en/stable/reference/samplers.html."
            )

        self._crossover_prob = crossover_prob
        self._mutation_prob = mutation_prob
        self._swapping_prob = swapping_prob
        self._mutation = mutation
        self._crossover = crossover
        self._constraints_func = constraints_func
        self._rng = rng

    def __call__(
        self,
        study: Study,
        search_space: dict[str, BaseDistribution],
        parent_population: list[FrozenTrial],
    ) -> dict[str, Any]:
        """Generate a child parameter from the given parent population by NSGA-II algorithm.
        Args:
            study:
                Target study object.
            search_space:
                A dictionary containing the parameter names and parameter's distributions.
            parent_population:
                A list of trials that are selected as parent population.
        Returns:
            A dictionary containing the parameter names and parameter's values.
        """
        if _is_constrained_optimization(parent_population):
            dominates = _constrained_dominates
        else:
            dominates = _dominates
        # We choose a child based on the specified crossover method.
        if self._rng.rng.rand() < self._crossover_prob:
            child_params = perform_crossover(
                self._crossover,
                study,
                parent_population,
                search_space,
                self._rng.rng,
                self._swapping_prob,
                dominates,
            )
        else:
            parent_population_size = len(parent_population)
            parent_params = parent_population[self._rng.rng.choice(parent_population_size)].params
            child_params = {name: parent_params[name] for name in search_space.keys()}

        n_params = len(child_params)
        if self._mutation_prob is None:
            mutation_prob = 1.0 / max(1.0, n_params)
        else:
            mutation_prob = self._mutation_prob

        params = {}
        for param_name in child_params.keys():
            if self._rng.rng.rand() >= mutation_prob:
                params[param_name] = child_params[param_name]
            elif self._mutation is not None:
                mutation_value = perform_mutation(
                    self._mutation,
                    self._rng.rng,
                    study,
                    search_space[param_name],
                    child_params[param_name],
                )
                if mutation_value is not None:
                    params[param_name] = mutation_value
        return params
