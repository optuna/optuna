import abc
from typing import Dict

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.study import Study


class BaseCrossover(object, metaclass=abc.ABCMeta):
    """Base class for crossovers.

    A crossover operation is used by :class:`~optuna.samplers.NSGAIISampler`
    to create new parameter combination from parameters of `n` parent individuals.
    """

    def __str__(self) -> str:

        return self.__class__.__name__

    @property
    @abc.abstractmethod
    def n_parents(self) -> int:
        """Number of parent individuals required to perform crossover."""

        raise NotImplementedError

    @abc.abstractmethod
    def crossover(
        self,
        parents_params: np.ndarray,
        rng: np.random.RandomState,
        study: Study,
        search_space: Dict[str, BaseDistribution],
    ) -> np.ndarray:
        """Perform crossover of selected parent individuals.

        This method is called in :func:`~optuna.samplers.NSGAIISampler.sample_relative`.

        Args:
            parents_params:
                A ``numpy.ndarray`` with dimensions ``num_parents x num_parameters``.
                Represents a continuous parameter space for each parent individual.
            rng:
                An instance of `numpy.random.RandomState`.
            study:
                Target study object.
            search_space:
                The search space returned by
                :func:`~optuna.samplers.BaseSampler.infer_relative_search_space`.

        Returns:
            A 1-dimensional ``numpy.ndarray`` containing new parameter combination.
        """

        raise NotImplementedError
