from __future__ import annotations

import abc
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence
import warnings

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.samplers.nsgaii._crossover import perform_crossover
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.samplers.nsgaii._crossovers._uniform import UniformCrossover
from optuna.samplers.nsgaii._elite_population_selection_strategy import _constrained_dominates
from optuna.study import Study
from optuna.study._multi_objective import _dominates
from optuna.trial import FrozenTrial


class NSGAIIChildGenerationStrategy:
    def __init__(
        self,
        *,
        population_size: int,
        crossover_prob: float,
        mutation_prob: float | None,
        swapping_prob: float,
        crossover: BaseCrossover,
        constraints_func: Optional[Callable[[FrozenTrial], Sequence[float]]] = None,
        seed: Optional[int] = None,
    ) -> None:
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
        self._crossover_prob = crossover_prob
        self._mutation_prob = mutation_prob
        self._swapping_prob = swapping_prob
        self._crossover = crossover
        self._constraints_func = constraints_func
        self._rng = np.random.RandomState(seed)

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
        dominates = _dominates if self._constraints_func is None else _constrained_dominates
        # We choose a child based on the specified crossover method.
        if self._rng.rand() < self._crossover_prob:
            child_params = perform_crossover(
                self._crossover,
                study,
                parent_population,
                search_space,
                self._rng,
                self._swapping_prob,
                dominates,
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
